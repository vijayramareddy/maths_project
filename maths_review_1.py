import numpy as np
import math, heapq, sys, io

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace', line_buffering=True)
except Exception:
    pass
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
from matplotlib.animation import FuncAnimation
import time
import argparse
import os
import csv


# ══════════════════════════════════════════════════════════════════════════════
#  MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

def d2(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ══════════════════════════════════════════════════════════════════════════════
#  OBSTACLE
# ══════════════════════════════════════════════════════════════════════════════
class Obstacle:
    def __init__(self, x, y, r=0.5):
        self.x, self.y, self.r = float(x), float(y), float(r)


# ══════════════════════════════════════════════════════════════════════════════
#  GRID MAP
# ══════════════════════════════════════════════════════════════════════════════
class GridMap:
    RES = 0.2

    def __init__(self):
        n = math.ceil(10.0 / self.RES)
        self.N = n
        self.cells = np.zeros((n, n), dtype=np.uint8)

    def w2g(self, x, y):
        return (clamp(int(x / self.RES), 0, self.N - 1),
                clamp(int(y / self.RES), 0, self.N - 1))

    def g2w(self, i, j):
        return (i * self.RES + self.RES / 2,
                j * self.RES + self.RES / 2)

    def free(self, i, j):
        return 0 <= i < self.N and 0 <= j < self.N and self.cells[i, j] == 0

    def inflate(self, obstacles, margin=0.40):
        self.cells[:] = 0
        for o in obstacles:
            r = o.r + margin
            ci, cj = self.w2g(o.x, o.y)
            span = int(r / self.RES) + 2
            for di in range(-span, span + 1):
                for dj in range(-span, span + 1):
                    ni, nj = ci + di, cj + dj
                    if not (0 <= ni < self.N and 0 <= nj < self.N):
                        continue
                    wx, wy = self.g2w(ni, nj)
                    if d2(wx, wy, o.x, o.y) <= r:
                        self.cells[ni, nj] = 1

    def los(self, x1, y1, x2, y2):
        dist = d2(x1, y1, x2, y2)
        steps = max(3, int(dist / (self.RES * 0.35)))
        for s in range(steps + 1):
            t = s / steps
            gi, gj = self.w2g(x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            if not self.free(gi, gj):
                return False
        return True

    def safe(self, i, j):
        if self.free(i, j):
            return (i, j)
        for r in range(1, 12):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if abs(di) == r or abs(dj) == r:
                        ni, nj = i + di, j + dj
                        if self.free(ni, nj):
                            return (ni, nj)
        return (i, j)


# ══════════════════════════════════════════════════════════════════════════════
#  PATH SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════
def smooth(pts, gmap):
    if not pts or len(pts) < 2:
        return pts
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        farthest = i + 1
        for j in range(len(pts) - 1, i + 1, -1):
            if gmap.los(*pts[i], *pts[j]):
                farthest = j
                break
        out.append(pts[farthest])
        i = farthest
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  A*
# ══════════════════════════════════════════════════════════════════════════════
def plan_astar(gmap, sx, sy, gx, gy):
    si, sj = gmap.safe(*gmap.w2g(sx, sy))
    gi, gj = gmap.safe(*gmap.w2g(gx, gy))

    def h(i, j):
        return d2(i, j, gi, gj) * gmap.RES

    key = lambda i, j: i * 10000 + j
    g_sc = {key(si, sj): 0.0}
    came = {}
    ctr  = 0
    heap = [(h(si, sj), ctr, si, sj)]

    for _ in range(30000):
        if not heap:
            break
        f, _, ci, cj = heapq.heappop(heap)
        ck = key(ci, cj)
        if f > g_sc.get(ck, 1e18) + h(ci, cj) + 1e-9:
            continue
        if ci == gi and cj == gj:
            raw = []
            cur = ck
            while cur in came:
                pi, pj = came[cur]
                raw.append(gmap.g2w(pi, pj))
                cur = key(pi, pj)
            raw.reverse()
            raw.append((gx, gy))
            return smooth(raw, gmap)
        cg = g_sc[ck]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = ci + di, cj + dj
                if not gmap.free(ni, nj):
                    continue
                ng = cg + d2(di, dj, 0, 0) * gmap.RES
                nk = key(ni, nj)
                if ng < g_sc.get(nk, 1e18):
                    g_sc[nk] = ng
                    came[nk] = (ci, cj)
                    ctr += 1
                    heapq.heappush(heap, (ng + h(ni, nj), ctr, ni, nj))
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════
ALGO_NAMES  = {1: "A*"}
ALGO_COLORS = {1: "#00ff99"}

def compute_path(algo_id, gmap, obstacles, sx, sy, gx, gy):
    p = plan_astar(gmap, sx, sy, gx, gy)
    if not p:
        n = 12
        p = [(sx + (gx - sx) * s / n, sy + (gy - sy) * s / n)
             for s in range(1, n + 1)]
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  MLP  8 -> 16 -> 12 -> 2
# ══════════════════════════════════════════════════════════════════════════════
class MLP:
    ARCH = [8, 16, 12, 2]

    def __init__(self):
        np.random.seed(42)
        self.W, self.b       = [], []
        self.velW, self.velb = [], []
        self.t               = 0
        self.lr              = 0.005
        self.momentum        = 0.9
        self.enabled         = False
        self.updates         = 0
        self.violations      = 0
        self.lr_log          = []
        self.loss_log        = []   # per-update stability loss (for post-sim plot)
        self._ema            = np.zeros(2)
        self._ema_beta       = 0.85

        for i in range(len(self.ARCH) - 1):
            fi, fo = self.ARCH[i], self.ARCH[i + 1]
            sc = math.sqrt(2.0 / fi)
            self.W.append(np.random.randn(fi, fo) * sc)
            self.b.append(np.zeros(fo))
            self.velW.append(np.zeros((fi, fo)))
            self.velb.append(np.zeros(fo))

    def forward(self, x):
        if not self.enabled:
            return np.zeros(2)
        a = np.asarray(x, dtype=float)
        for l in range(len(self.W) - 1):
            a = np.maximum(0, a @ self.W[l] + self.b[l])
        zL  = a @ self.W[-1] + self.b[-1]
        out = 0.5 * np.tanh(zL)
        self._ema = self._ema_beta * self._ema + (1 - self._ema_beta) * out
        return self._ema.copy()

    def backward(self, x, dVdt, obs_viol):
        if not self.enabled or self.updates > 8000:
            return
        self.t += 1
        a    = np.asarray(x, dtype=float)
        acts = [a.copy()]
        zs   = []
        for l in range(len(self.W) - 1):
            z = a @ self.W[l] + self.b[l]
            zs.append(z)
            a = np.maximum(0, z)
            acts.append(a.copy())
        zL  = a @ self.W[-1] + self.b[-1]
        zs.append(zL)
        out = 0.5 * np.tanh(zL)
        acts.append(out.copy())

        oG = np.zeros(2)
        if dVdt > 0:
            severity = min(dVdt, 5.0)
            oG += 3.0 * severity * np.sign(out)
            self.violations += 1
        else:
            oG -= 0.03 * out
        if obs_viol > 0:
            oG += 2.5 * np.sign(out) * obs_viol

        delta = oG * 0.5 * (1 - np.tanh(zL) ** 2)
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        gW[-1] = np.outer(acts[-2], delta)
        gb[-1] = delta.copy()

        for l in range(len(self.W) - 2, -1, -1):
            drelu = (zs[l] > 0).astype(float)
            delta = drelu * (delta @ self.W[l + 1].T)
            gW[l] = np.outer(acts[l], delta)
            gb[l] = delta.copy()

        gnorm = math.sqrt(sum(float(np.sum(g ** 2)) for g in gW)) or 1.0
        sc    = min(1.0, 0.5 / gnorm)

        for l in range(len(self.W)):
            gw = gW[l] * sc
            self.velW[l] = self.momentum * self.velW[l] + gw
            self.W[l]   -= self.lr * self.velW[l]
            gb2 = gb[l] * sc
            self.velb[l] = self.momentum * self.velb[l] + gb2
            self.b[l]   -= self.lr * self.velb[l]

        # scalar stability loss = ||oG||^2  (measures how far from stability)
        step_loss = float(np.sum(oG ** 2))
        self.loss_log.append(step_loss)

        if   dVdt > 0.2:   self.lr = min(self.lr * 1.5,  0.015)
        elif dVdt > 0.0:   self.lr = min(self.lr * 1.15, 0.015)
        elif dVdt > -0.05: pass
        else:               self.lr = max(self.lr * 0.96, 5e-5)
        self.lr_log.append(self.lr)
        self.updates += 1


# ══════════════════════════════════════════════════════════════════════════════
#  LYAPUNOV
# ══════════════════════════════════════════════════════════════════════════════
_P = np.array([1.5, 1.5, 0.8, 0.5, 2.0])

def lyap_V(e):
    return 0.5 * float(np.dot(_P * e, e))


# ══════════════════════════════════════════════════════════════════════════════
#  ROBOT DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════
class Robot:
    MASS    = 10.0
    INERTIA = 0.6
    DRAG    = 0.25

    def __init__(self, x, y):
        self.x     = float(x)
        self.y     = float(y)
        self.theta = math.pi / 4
        self.v     = 0.0
        self.w     = 0.0

    def step(self, F, T, dt):
        F = clamp(F, -45.0, 45.0)
        T = clamp(T, -9.0,   9.0)
        self.v += ((F / self.MASS)    - self.DRAG * self.v) * dt
        self.w += ((T / self.INERTIA) - self.DRAG * self.w) * dt
        self.v  = clamp(self.v, -2.2, 2.2)
        self.w  = clamp(self.w, -4.0, 4.0)
        self.theta = wrap(self.theta + self.w * dt)
        self.x = clamp(self.x + self.v * math.cos(self.theta) * dt, 0.2, 9.8)
        self.y = clamp(self.y + self.v * math.sin(self.theta) * dt, 0.2, 9.8)


# ══════════════════════════════════════════════════════════════════════════════
#  CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════
class Controller:
    R_WP        = 0.35
    V_CRUISE    = 1.0
    V_SLOW      = 0.5
    OBS_SLOW_D  = 1.0
    OBS_REP_D   = 1.5
    GOAL_TOL    = 0.50
    K_CTE       = 4.0

    def __init__(self, obstacles, goal, start, algo_id):
        self.obstacles = obstacles          # shared reference – updated live
        self.goal      = (float(goal[0]), float(goal[1]))
        self.start     = (float(start[0]), float(start[1]))
        self.algo_id   = algo_id
        self.robot     = Robot(*start)
        self.nn        = MLP()
        self.gmap      = GridMap()
        self.gmap.inflate(obstacles)
        self.collect_data_flag = True

        self.path      = []
        self.wp_idx    = 0
        self.traj      = []
        self.hist      = {k: [] for k in
                         ('t','V','dVdt','lr','nn0','nn1','margin','gd','od','spd',
                          'x','y','wx','wy','v_des','theta','theta_des','ups','enorm')}
        self.collected_samples = []
        self.last_wall_time    = time.time()
        self._prev_V           = None
        self._stk_t            = 0.0
        self._chk_pos          = list(start)
        self._chk_t            = 0.0
        self.live_replans      = 0

        self.replan()

    def replan(self):
        """Replan from current robot position. Safe to call at any time."""
        self.gmap.inflate(self.obstacles)
        r = self.robot
        self.path   = compute_path(self.algo_id, self.gmap, self.obstacles,
                                   r.x, r.y, *self.goal)
        self.wp_idx = 0

    def _wp(self):
        if self.wp_idx < len(self.path):
            return self.path[self.wp_idx]
        return self.goal

    def _min_od(self):
        r  = self.robot
        md = 1e9
        for o in self.obstacles:
            dd = d2(r.x, r.y, o.x, o.y) - o.r
            if dd < md:
                md = dd
        return md

    def _cross_track_error(self):
        r = self.robot
        if self.wp_idx > 0 and self.wp_idx < len(self.path):
            ax, ay = self.path[self.wp_idx - 1]
            bx, by = self.path[self.wp_idx]
        elif self.wp_idx == 0 and len(self.path) > 0:
            ax, ay = self.start
            bx, by = self.path[0]
        else:
            return 0.0
        dx, dy  = bx - ax, by - ay
        seg_len = d2(ax, ay, bx, by)
        if seg_len < 0.01:
            return 0.0
        cross = dx * (r.y - ay) - dy * (r.x - ax)
        return cross / seg_len

    def step(self, dt, t):
        now = time.time()
        dt_wall = now - self.last_wall_time
        self.last_wall_time = now
        ups = 1.0 / dt_wall if dt_wall > 1e-4 else 0.0

        r = self.robot

        wx, wy = self._wp()
        if d2(r.x, r.y, wx, wy) < self.R_WP:
            if self.wp_idx < len(self.path) - 1:
                self.wp_idx += 1
            wx, wy = self._wp()

        self._chk_t += dt
        if self._chk_t >= 1.0:
            moved = d2(r.x, r.y, *self._chk_pos)
            if moved < 0.12:
                self._stk_t += 1.0
            else:
                self._stk_t = 0.0
            self._chk_pos = [r.x, r.y]
            self._chk_t   = 0.0

        if self._stk_t >= 3.0:
            if self.wp_idx < len(self.path) - 1:
                self.wp_idx += 1
            self.replan()
            self._stk_t = 0.0

        twx, twy = self._wp()
        exG = twx - r.x
        eyG = twy - r.y
        dTW = d2(r.x, r.y, twx, twy)
        goal_d = d2(r.x, r.y, *self.goal)
        min_od = self._min_od()
        e_ct   = self._cross_track_error()

        cos_t    = math.cos(r.theta)
        sin_t    = math.sin(r.theta)
        ex       =  cos_t * exG + sin_t * eyG
        ey       = -sin_t * exG + cos_t * eyG
        desTheta = math.atan2(eyG, exG)
        etheta   = wrap(desTheta - r.theta)

        if min_od < self.OBS_SLOW_D:
            v_des = self.V_SLOW
        elif goal_d < 1.5:
            v_des = max(0.3, goal_d * 0.4)
        else:
            v_des = self.V_CRUISE
        ev = v_des - r.v

        err = np.array([ex, ey, etheta, ev, e_ct])
        V   = lyap_V(err)

        if t > 1.5 and not self.nn.enabled:
            self.nn.enabled = True

        nn_in  = np.array([ex, ey, etheta, ev, min_od, dTW, goal_d, e_ct])
        nn_out = self.nn.forward(nn_in)

        if self.collect_data_flag:
            self.collected_samples.append(np.concatenate([nn_in, nn_out]))

        KP_V, KD_V = 8.0, 4.0
        KP_H, KD_H = 14.0, 5.0

        a_base     = KP_V * ev       - KD_V * (r.v - v_des)
        alpha_base = KP_H * etheta   - KD_H * r.w
        alpha_base -= self.K_CTE * e_ct

        aT = a_base     + nn_out[0]
        aA = alpha_base + nn_out[1]

        if min_od < self.OBS_REP_D:
            for o in self.obstacles:
                dx  = r.x - o.x
                dy  = r.y - o.y
                dd  = d2(r.x, r.y, o.x, o.y) - o.r
                if 0.05 < dd < self.OBS_REP_D:
                    lat  = -sin_t * dx + cos_t * dy
                    mag  = 3.5 * math.exp(-dd / 0.5)
                    aA  += mag * lat

        if min_od < 0.15:
            aT = -10.0 * r.v
            aA = -6.0  * r.w

        r.step(r.MASS * aT, r.INERTIA * aA, dt)

        dVdt   = 0.0
        margin = 0.0
        if self._prev_V is not None:
            dVdt = (V - self._prev_V) / dt
            if abs(dVdt) < 50:
                obs_viol = max(0.0, 0.35 - min_od)
                self.nn.backward(nn_in, dVdt, obs_viol)
                margin = -(dVdt + 0.18 * V)
        self._prev_V = V

        self.traj.append((r.x, r.y))
        h = self.hist
        h['t'].append(t);     h['V'].append(V);      h['dVdt'].append(dVdt)
        h['lr'].append(self.nn.lr)
        h['nn0'].append(float(nn_out[0])); h['nn1'].append(float(nn_out[1]))
        h['margin'].append(margin)
        h['x'].append(r.x); h['y'].append(r.y)
        h['wx'].append(twx); h['wy'].append(twy)
        h['v_des'].append(v_des)
        h['theta'].append(r.theta); h['theta_des'].append(desTheta)
        h['gd'].append(goal_d); h['od'].append(min_od); h['spd'].append(r.v)
        h['ups'].append(ups); h['enorm'].append(np.linalg.norm(err))

        return goal_d < self.GOAL_TOL


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
class Simulation:
    START = (1.0, 1.0)
    GOAL  = (9.0, 9.0)
    DT    = 0.04

    def __init__(self, collect_path=None, auto_start=False):
        self.obstacles    = []
        self.algo_id      = 1
        self.obs_r        = 0.5
        self.ctrl         = None
        self.phase        = "place"
        self.collect_path = collect_path
        self.t            = 0.0
        self.anim         = None
        # ── live replan state ──────────────────────────────────
        self._replan_flash  = 0
        self._flash_text    = None
        self._replan_count  = 0
        self._build_ui()
        if auto_start:
            self.start_sim()

    # ── BUILD UI ──────────────────────────────────────────────
    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 9), facecolor="#050c18")
        gs = self.fig.add_gridspec(3, 4, hspace=0.38, wspace=0.3,
                                   left=0.04, right=0.97, top=0.94, bottom=0.06)
        self.ax  = self.fig.add_subplot(gs[:, :2])
        self.axV = self.fig.add_subplot(gs[0, 2])
        self.axD = self.fig.add_subplot(gs[0, 3])
        self.axL = self.fig.add_subplot(gs[1, 2])
        self.axN = self.fig.add_subplot(gs[1, 3])
        self.axM = self.fig.add_subplot(gs[2, 2])
        self.axS = self.fig.add_subplot(gs[2, 3])

        BG = "#060e1c"
        for a in (self.ax, self.axV, self.axD, self.axL, self.axN, self.axM):
            a.set_facecolor(BG)
            for sp in a.spines.values():
                sp.set_color("#1a3050")
            a.tick_params(colors="#3a5878", labelsize=7)

        self.ax.set_xlim(-0.3, 10.3)
        self.ax.set_ylim(-0.3, 10.3)
        self.ax.set_aspect('equal')
        self.ax.grid(True, color="#0d1e32", lw=0.6, alpha=0.7)
        self.ax.set_xlabel("X (m)", color="#3a5878", fontsize=10)
        self.ax.set_ylabel("Y (m)", color="#3a5878", fontsize=10)

        self.ax.add_patch(Circle(self.START, 0.28, color="#00ff77", zorder=6,
                                 ec="#004422", lw=2))
        self.ax.text(*self.START, "S", ha='center', va='center',
                     fontsize=9, fontweight='bold', color='black', zorder=7)
        self.ax.add_patch(Circle(self.GOAL, 0.28, color="#ffd700", zorder=6,
                                 ec="#997700", lw=2))
        self.ax.text(*self.GOAL, "G", ha='center', va='center',
                     fontsize=9, fontweight='bold', color='black', zorder=7)
        for rr in (0.5, 0.85, 1.2):
            self.ax.add_patch(Circle(self.GOAL, rr, fill=False,
                                     ec="#ffd70028", lw=1.2, zorder=5))

        self._traj,  = self.ax.plot([], [], '-', color="#3dd0ff", lw=2, zorder=4)
        self._path,  = self.ax.plot([], [], '--', lw=2.2, zorder=3,
                                    color="#00ff99", dashes=(8, 5))
        self._rbody  = Circle((0, 0), 0.30, color="#1a3a77", zorder=9,
                               ec="#00ccff", lw=2.5)
        self._rarrow = FancyArrow(0, 0, 0.5, 0, width=0.16,
                                  head_width=0.32, head_length=0.18,
                                  fc="#00ffff", ec="#001144", zorder=10)
        self.ax.add_patch(self._rbody)
        self.ax.add_patch(self._rarrow)

        # ── live replan flash label (new) ──────────────────────
        self._flash_text = self.ax.text(
            5.0, 9.7, "", ha='center', va='top', fontsize=11,
            fontweight='bold', color="#ff6600",
            bbox=dict(boxstyle='round,pad=0.4', fc="#1a0a00", ec="#ff6600", lw=2),
            zorder=25, alpha=0.0)

        def mk(a, col, ttl):
            ln, = a.plot([], [], color=col, lw=1.8)
            a.set_title(ttl, color="#4a9ac0", fontsize=8, pad=3)
            a.axhline(0, color="#ff443355", lw=0.8, ls='--')
            return ln

        self.lnV  = mk(self.axV, "#c080ff", "Lyapunov V(e)")
        self.lnD  = mk(self.axD, "#ff9922", "dV/dt  (stability)")
        self.lnL  = mk(self.axL, "#4488ff", "Learning Rate")
        self.lnN  = mk(self.axN, "#ff4488", "NN Delta Force")
        self.lnMg = mk(self.axM, "#44ff88", "Stability Margin")

        self.axS.set_facecolor("#060e1c")
        for sp in self.axS.spines.values():
            sp.set_color("#1a3050")
        self.axS.axis('off')
        self._stxt = self.axS.text(0.05, 0.97, "",
                                   transform=self.axS.transAxes,
                                   va='top', fontsize=7.5, family='monospace',
                                   color="#6090b8",
                                   bbox=dict(boxstyle='round', fc="#07111f",
                                             ec="#162840", lw=1))
        self._itxt = self.ax.text(0.02, 0.99, self._info(),
                                  transform=self.ax.transAxes, va='top',
                                  fontsize=8, family='monospace', color="#4af0ff",
                                  bbox=dict(boxstyle='round', fc="#07111f99",
                                            ec="#1a3050", lw=1), zorder=20)
        self._title = self.fig.suptitle(
            f"ADAPTIVE NEURAL LYAPUNOV  .  {ALGO_NAMES[self.algo_id]}",
            color="#4af0ff", fontsize=13, fontweight='bold', y=0.98)

        self._obs_art = []
        self._wpdots  = []

        self.fig.canvas.mpl_connect('button_press_event', self._click)
        self.fig.canvas.mpl_connect('key_press_event',    self._key)

    # ── INFO TEXT ─────────────────────────────────────────────
    def _info(self):
        aname = ALGO_NAMES[self.algo_id]
        if self.phase == "place":
            return (f"PLACE OBSTACLES\n"
                    f"------------------\n"
                    f"Algo : {aname}\n"
                    f"r    : {self.obs_r:.1f} m\n"
                    f"Obs  : {len(self.obstacles)}\n"
                    f"------------------\n"
                    f"Click  : add\n"
                    f"RClick : remove\n"
                    f"SPACE  : START\n"
                    f"1      : algo\n"
                    f"+/-    : radius\n"
                    f"R      : reset")
        wp  = self.ctrl.wp_idx if self.ctrl else 0
        nwp = len(self.ctrl.path) if self.ctrl else 0
        return (f"{'RUNNING' if self.phase=='run' else 'DONE'}\n"
                f"------------------\n"
                f"Algo    : {aname}\n"
                f"t       : {self.t:.1f} s\n"
                f"WP      : {wp}/{nwp}\n"
                f"Replans : {self._replan_count}\n"
                f"------------------\n"
                f"Click  : live obs\n"
                f"RClick : remove\n"
                f"P      : replan\n"
                f"R      : reset")

    def _stats(self):
        c = self.ctrl
        if not c:
            return (f"Algorithm : {ALGO_NAMES[self.algo_id]}\n"
                    f"NN        : standby\n\n"
                    f"Architecture:\n8->16->12->2\nReLU+tanh\n\n"
                    f"Lyapunov:\nV=0.5*e'Pe + CTE\nP=diag(1.5,..,2.0)\nCond: dV/dt<0")
        h = c.hist
        def L(k): return h[k][-1] if h[k] else 0.0
        sp = (100 * sum(1 for v in h['dVdt'] if v < 0) / max(len(h['dVdt']), 1))
        return (f"-- RUNTIME --\n"
                f"Algo    : {ALGO_NAMES[self.algo_id]}\n"
                f"NN      : {'ACTIVE' if c.nn.enabled else 'standby'}\n"
                f"t       : {self.t:.1f} s\n"
                f"V(e)    : {L('V'):.4f}\n"
                f"dV/dt   : {L('dVdt'):.4f}\n"
                f"Goal d  : {L('gd'):.2f} m\n"
                f"Obs d   : {L('od'):.2f} m\n"
                f"Speed   : {L('spd'):.2f} m/s\n"
                f"--\n"
                f"Updates : {c.nn.updates}\n"
                f"Violat  : {c.nn.violations}\n"
                f"Stable  : {sp:.0f}%\n"
                f"LR      : {c.nn.lr:.2e}\n"
                f"Replans : {self._replan_count}\n"
                f"--\n"
                f"V=0.5*e'Pe+CTE  P=diag(1.5,..,2.0)\n"
                f"8->16->12->2 ReLU+tanh\n"
                f"Stability-Aware Adaptive SGD")

    # ── OBSTACLE DRAWING ──────────────────────────────────────
    def _draw_obs(self):
        for bp, rp in self._obs_art:
            bp.remove(); rp.remove()
        self._obs_art = []
        for o in self.obstacles:
            bp = Circle((o.x, o.y), o.r, color="#aa2020",
                        ec="#ff3333", lw=2, zorder=4, alpha=0.85)
            rp = Circle((o.x, o.y), o.r + 0.40, fill=False,
                        ec="#ff222218", lw=1, ls='--', zorder=3)
            self.ax.add_patch(bp)
            self.ax.add_patch(rp)
            self._obs_art.append((bp, rp))

    # ── LIVE REPLAN HELPER (new) ───────────────────────────────
    def _live_replan(self, label: str):
        """
        Called immediately after a live obstacle add/remove during "run".
        1. Tells the controller to rebuild the grid map with new obstacles.
        2. Replans the path from the robot's current position.
        3. Shows a brief orange flash label on the arena canvas.
        4. Updates the sidebar counters.
        """
        self.ctrl.replan()
        self._replan_count += 1
        self._replan_flash = 8                     # ~320 ms at 40 ms/frame
        self._flash_text.set_text(
            f"LIVE REPLAN #{self._replan_count}  [{label}]")
        self._flash_text.set_alpha(1.0)
        self._refresh_path()
        self._itxt.set_text(self._info())
        self._stxt.set_text(self._stats())
        self.fig.canvas.draw_idle()
        print(f">> Live replan #{self._replan_count} ({label})  "
              f"path={len(self.ctrl.path)} wps  t={self.t:.1f}s")

    # ── START ─────────────────────────────────────────────────
    def start_sim(self):
        self.phase         = "run"
        self.t             = 0.0
        self._replan_count = 0
        self.ctrl = Controller(self.obstacles, self.GOAL,
                               self.START, self.algo_id)
        self._path.set_color(ALGO_COLORS[self.algo_id])
        self._refresh_path()
        self._title.set_text(
            f"NEURAL LYAPUNOV  .  {ALGO_NAMES[self.algo_id]}  .  RUNNING")
        print(f"\n> {ALGO_NAMES[self.algo_id]}  --  {len(self.ctrl.path)} waypoints")
        for p in self.ctrl.path:
            print(f"   {tuple(round(v,2) for v in p)}")
        self.anim = FuncAnimation(self.fig, self._animate,
                                  interval=40, blit=False, repeat=False,
                                  cache_frame_data=False)

    def _refresh_path(self):
        c = self.ctrl
        for d in self._wpdots:
            d.remove()
        self._wpdots = []
        if c and c.path:
            xs = [p[0] for p in c.path]
            ys = [p[1] for p in c.path]
            self._path.set_data(xs, ys)
            for i, (px, py) in enumerate(c.path):
                active = (i == c.wp_idx)
                dot = Circle((px, py),
                             0.10 if active else 0.05,
                             color="#00ffcc" if active else "#ffffff44",
                             zorder=5)
                self.ax.add_patch(dot)
                self._wpdots.append(dot)
        else:
            self._path.set_data([], [])

    # ── ANIMATION ─────────────────────────────────────────────
    def _animate(self, _frame):
        c = self.ctrl
        if not c:
            return

        done = c.step(self.DT, self.t)
        self.t += self.DT

        r = c.robot
        self._rbody.center = (r.x, r.y)
        self._rarrow.remove()
        dx = 0.50 * math.cos(r.theta)
        dy = 0.50 * math.sin(r.theta)
        self._rarrow = FancyArrow(r.x, r.y, dx, dy,
                                  width=0.16, head_width=0.32,
                                  head_length=0.18,
                                  fc="#00ffff", ec="#001144", zorder=10)
        self.ax.add_patch(self._rarrow)

        if c.traj:
            tx = [p[0] for p in c.traj[-400:]]
            ty = [p[1] for p in c.traj[-400:]]
            self._traj.set_data(tx, ty)

        self._refresh_path()

        h = c.hist
        if len(h['t']) > 1:
            ts = h['t']
            for ax_, ln, dat in [
                (self.axV, self.lnV,  h['V']),
                (self.axD, self.lnD,  h['dVdt']),
                (self.axL, self.lnL,  h['lr']),
                (self.axN, self.lnN,  h['nn0']),
                (self.axM, self.lnMg, h['margin']),
            ]:
                ln.set_data(ts, dat)
                ax_.relim(); ax_.autoscale_view()

        # ── tick flash countdown (new) ─────────────────────────
        if self._replan_flash > 0:
            self._replan_flash -= 1
            if self._replan_flash == 0:
                self._flash_text.set_alpha(0.0)

        self._stxt.set_text(self._stats())
        self._itxt.set_text(self._info())

        if done or self.t > 90:
            if self.anim:
                self.anim.event_source.stop()
            self.phase = "done"
            label = "SUCCESS" if done else "TIME LIMIT"
            self._title.set_text(
                f"{ALGO_NAMES[self.algo_id]}  .  {label}  ({self.t:.1f}s)  "
                f".  Press R to reset")
            print(f"\n{'[OK]' if done else '[TIME]'} {label} at t={self.t:.1f}s  "
                  f"live_replans={self._replan_count}")
            self._itxt.set_text(self._info())

    # ── INPUT EVENTS ──────────────────────────────────────────
    def _click(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None:
            return
        x, y = ev.xdata, ev.ydata

        # ── PLACE PHASE: unchanged behaviour ──────────────────
        if self.phase == "place":
            if ev.button == 1:
                if d2(x, y, *self.START) < 1.1: return
                if d2(x, y, *self.GOAL)  < 1.1: return
                self.obstacles.append(Obstacle(x, y, self.obs_r))
                self._draw_obs()
                self._itxt.set_text(self._info())
                self.fig.canvas.draw_idle()
            elif ev.button == 3:
                idx = -1; bd = 1e9
                for i, o in enumerate(self.obstacles):
                    dd = d2(x, y, o.x, o.y)
                    if dd < bd and dd < o.r + 0.5:
                        bd, idx = dd, i
                if idx >= 0:
                    self.obstacles.pop(idx)
                    self._draw_obs()
                    self._itxt.set_text(self._info())
                    self.fig.canvas.draw_idle()

        # ── RUN PHASE: live obstacle add / remove (new) ────────
        elif self.phase == "run" and self.ctrl is not None:
            r = self.ctrl.robot

            if ev.button == 1:
                # Guard: don't place on robot body or goal zone
                if d2(x, y, r.x, r.y) < 0.6:
                    print(">> Blocked: cannot place obstacle on robot")
                    return
                if d2(x, y, *self.GOAL) < 0.8:
                    print(">> Blocked: cannot place obstacle on goal")
                    return
                # Add to shared list – controller already holds a reference
                self.obstacles.append(Obstacle(x, y, self.obs_r))
                self._draw_obs()
                self._live_replan(f"+obs@({x:.1f},{y:.1f})")

            elif ev.button == 3:
                # Remove nearest obstacle within click tolerance
                idx = -1; bd = 1e9
                for i, o in enumerate(self.obstacles):
                    dd = d2(x, y, o.x, o.y)
                    if dd < bd and dd < o.r + 0.5:
                        bd, idx = dd, i
                if idx >= 0:
                    rm = self.obstacles.pop(idx)
                    self._draw_obs()
                    self._live_replan(f"-obs@({rm.x:.1f},{rm.y:.1f})")

    def _key(self, ev):
        k = ev.key

        if k == ' ' and self.phase == "place":
            self.start_sim()

        elif k == 'r':
            self.reset()

        elif k == 'p' and self.phase == "run" and self.ctrl:
            self.ctrl.replan()
            self._replan_count += 1
            self._refresh_path()
            print(f">> Manual replan #{self._replan_count}")

        elif k == '1':
            new_algo = 1
            if new_algo == self.algo_id and self.phase == "place":
                return
            self.algo_id = new_algo
            self._path.set_color(ALGO_COLORS[self.algo_id])
            print(f"Algorithm -> {ALGO_NAMES[self.algo_id]}")

            if self.phase == "place":
                self._title.set_text(
                    f"ADAPTIVE NEURAL LYAPUNOV  .  {ALGO_NAMES[self.algo_id]}")
                self._itxt.set_text(self._info())
                self._stxt.set_text(self._stats())
                self.fig.canvas.draw_idle()

            elif self.phase == "run" and self.ctrl:
                self.ctrl.algo_id = self.algo_id
                self.ctrl.replan()
                self._replan_count += 1
                self._refresh_path()
                self._title.set_text(
                    f"NEURAL LYAPUNOV  .  {ALGO_NAMES[self.algo_id]}  .  RUNNING")
                self._itxt.set_text(self._info())
                self._stxt.set_text(self._stats())
                self.fig.canvas.draw_idle()
                print(f">> Replanned with {ALGO_NAMES[self.algo_id]}")

            elif self.phase == "done":
                self.reset()
                self.start_sim()

        elif k in ('+', '=') and self.phase == "place":
            self.obs_r = min(1.5, round(self.obs_r + 0.1, 1))
            self._itxt.set_text(self._info())
            self.fig.canvas.draw_idle()

        elif k == '-' and self.phase == "place":
            self.obs_r = max(0.2, round(self.obs_r - 0.1, 1))
            self._itxt.set_text(self._info())
            self.fig.canvas.draw_idle()

    # ── RESET ─────────────────────────────────────────────────
    def reset(self):
        if self.anim is not None:
            try: self.anim.event_source.stop()
            except Exception: pass
            try: del self.anim
            except Exception: pass
            self.anim = None

        self.ctrl          = None
        self.phase         = "place"
        self.t             = 0.0
        self._replan_count = 0
        self._replan_flash = 0
        self._flash_text.set_alpha(0.0)

        self._traj.set_data([], [])
        self._path.set_data([], [])
        for d in self._wpdots:
            try: d.remove()
            except Exception: pass
        self._wpdots = []

        for ln, a in [(self.lnV, self.axV), (self.lnD, self.axD),
                      (self.lnL, self.axL), (self.lnN, self.axN),
                      (self.lnMg, self.axM)]:
            ln.set_data([], [])
            a.relim(); a.autoscale_view()

        self._rbody.center = self.START
        try: self._rarrow.remove()
        except Exception: pass
        self._rarrow = FancyArrow(*self.START, 0.5, 0,
                                  width=0.16, head_width=0.32, head_length=0.18,
                                  fc="#00ffff", ec="#001144", zorder=10)
        self.ax.add_patch(self._rarrow)

        self._draw_obs()
        self._itxt.set_text(self._info())
        self._stxt.set_text(self._stats())
        self._title.set_text(
            f"ADAPTIVE NEURAL LYAPUNOV  .  {ALGO_NAMES[self.algo_id]}")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        print(">> Reset")

    def run(self):
        plt.show()

    def save_collected_data(self):
        if not self.collect_path or not self.ctrl:
            return
        data = np.array(self.ctrl.collected_samples)
        if len(data) == 0:
            print("No training data collected.")
            return
        print(f"Saving {len(data)} samples to {self.collect_path}...")
        try:
            X = data[:, :8]
            Y = data[:, 8:]
            np.savez(self.collect_path, x=X, y=Y)
            print("Data saved successfully.")
        except Exception as e:
            print(f"Error saving data: {e}")

    def save_log_csv(self, filename="simulation_log.csv"):
        if not self.ctrl or not self.ctrl.hist:
            print("No simulation data to save.")
            return
        keys   = list(self.ctrl.hist.keys())
        length = len(self.ctrl.hist['t'])
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(keys)
                for i in range(length):
                    row = [self.ctrl.hist[k][i] for k in keys]
                    writer.writerow(row)
            print(f"Saved simulation log to {filename}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    # ─── POST-SIMULATION ANALYSIS PLOTS ──────────────────────
    def plot_results(self):
        """
        Four dedicated post-simulation figures:
          Fig 1 – Lyapunov Surface (3-D mesh of V over ex/ey grid)
          Fig 2 – Updates vs Time + Total Loss over updates
          Fig 3 – Position Tracking (x/y actual vs reference)
          Fig 4 – Velocity Tracking (actual vs desired speed)
        """
        if not self.ctrl or not self.ctrl.hist:
            print("No simulation data to plot.")
            return

        h  = self.ctrl.hist
        nn = self.ctrl.nn
        t  = np.asarray(h['t'])

        DARK   = "#07111f"
        GRID   = "#0d2035"
        SPINE  = "#1a3050"
        TC     = "#4af0ff"   # title colour
        MUTED  = "#3a6080"

        def _style(fig, axes_flat):
            fig.patch.set_facecolor(DARK)
            for ax in axes_flat:
                ax.set_facecolor("#060e1c")
                for sp in ax.spines.values():
                    sp.set_color(SPINE)
                ax.tick_params(colors=MUTED, labelsize=8)
                ax.xaxis.label.set_color(MUTED)
                ax.yaxis.label.set_color(MUTED)
                ax.title.set_color(TC)
                ax.grid(True, color=GRID, lw=0.7)

        # ══════════════════════════════════════════════════════
        # FIGURE 1 — LYAPUNOV SURFACE
        # V(ex, ey)  with  etheta=ev=e_ct=0 (slice through origin)
        # ══════════════════════════════════════════════════════
        fig1 = plt.figure(figsize=(13, 7), facecolor=DARK)
        fig1.suptitle("Lyapunov Surface  V(ex, ey)  [etheta=ev=ect=0]",
                      color=TC, fontsize=13, fontweight='bold', y=0.97)

        # ── left: 3-D surface ──
        ax3d = fig1.add_subplot(1, 2, 1, projection='3d')
        ax3d.set_facecolor(DARK)
        ax3d.tick_params(colors=MUTED, labelsize=7)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
            pane.set_edgecolor(SPINE)

        res   = 60
        ex_g  = np.linspace(-3.0, 3.0, res)
        ey_g  = np.linspace(-3.0, 3.0, res)
        EX, EY = np.meshgrid(ex_g, ey_g)
        VV  = np.zeros_like(EX)
        for i in range(res):
            for j in range(res):
                e = np.array([EX[i, j], EY[i, j], 0.0, 0.0, 0.0])
                VV[i, j] = lyap_V(e)

        surf = ax3d.plot_surface(EX, EY, VV,
                                 cmap='plasma', alpha=0.88,
                                 linewidth=0, antialiased=True)
        ax3d.set_xlabel("ex (m)",  color=MUTED, fontsize=8, labelpad=6)
        ax3d.set_ylabel("ey (m)",  color=MUTED, fontsize=8, labelpad=6)
        ax3d.set_zlabel("V(e)",    color=MUTED, fontsize=8, labelpad=6)
        ax3d.set_title("3-D Surface", color=TC, fontsize=10)
        cbar = fig1.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1)
        cbar.ax.tick_params(colors=MUTED, labelsize=7)
        cbar.ax.yaxis.label.set_color(MUTED)

        # ── right: contour + robot trajectory overlay ──
        ax2d = fig1.add_subplot(1, 2, 2)
        ax2d.set_facecolor("#060e1c")
        for sp in ax2d.spines.values():
            sp.set_color(SPINE)
        ax2d.tick_params(colors=MUTED, labelsize=8)
        ax2d.grid(True, color=GRID, lw=0.7)

        cf = ax2d.contourf(EX, EY, VV, levels=20, cmap='plasma', alpha=0.75)
        ax2d.contour(EX, EY, VV, levels=12, colors='white', linewidths=0.4, alpha=0.3)
        cbar2 = fig1.colorbar(cf, ax=ax2d, shrink=0.85)
        cbar2.ax.tick_params(colors=MUTED, labelsize=7)

        # overlay robot's (ex, ey) path through the Lyapunov landscape
        ex_traj = np.array(h.get('x', [])) - self.GOAL[0]
        ey_traj = np.array(h.get('y', [])) - self.GOAL[1]
        if len(ex_traj):
            sc = ax2d.scatter(ex_traj, ey_traj,
                              c=np.linspace(0, 1, len(ex_traj)),
                              cmap='cool', s=4, zorder=5, label='robot path')
            ax2d.plot(ex_traj[0],  ey_traj[0],  'go', ms=9, label='start', zorder=6)
            ax2d.plot(ex_traj[-1], ey_traj[-1], 'r*', ms=12, label='end',  zorder=6)
            ax2d.legend(fontsize=7, labelcolor=MUTED,
                        facecolor=DARK, edgecolor=SPINE)

        ax2d.set_xlabel("ex = x - gx  (m)", color=MUTED, fontsize=9)
        ax2d.set_ylabel("ey = y - gy  (m)", color=MUTED, fontsize=9)
        ax2d.set_title("Contour + Robot Error Trajectory", color=TC, fontsize=10)
        ax2d.set_xlim(-3.5, 3.5)
        ax2d.set_ylim(-3.5, 3.5)

        fig1.tight_layout(rect=[0, 0, 1, 0.95])

        # ══════════════════════════════════════════════════════
        # FIGURE 2 — NN UPDATES vs TIME  +  TOTAL LOSS vs UPDATES
        # ══════════════════════════════════════════════════════
        fig2, axs2 = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK)
        fig2.suptitle("Neural Network Learning Diagnostics",
                      color=TC, fontsize=13, fontweight='bold', y=0.97)
        _style(fig2, axs2.flat)

        # ── left: cumulative updates vs simulation time ──
        ax = axs2[0]
        # reconstruct update timestamps: nn activates at t>1.5, one update/step
        nn_start_idx = next((i for i, tv in enumerate(t) if tv > 1.5), 0)
        n_updates    = nn.updates
        upd_times    = t[nn_start_idx: nn_start_idx + n_updates] if n_updates else []
        upd_cumsum   = np.arange(1, len(upd_times) + 1)

        if len(upd_times):
            ax.plot(upd_times, upd_cumsum,
                    color="#44aaff", lw=2.0, label='cumulative updates')
            ax.fill_between(upd_times, upd_cumsum, alpha=0.15, color="#44aaff")
            # mark violations in red
            viol_mask = [v > 0 for v in h['dVdt'][nn_start_idx:
                                                   nn_start_idx + n_updates]]
            vt = [upd_times[i] for i, v in enumerate(viol_mask) if v]
            vc = [upd_cumsum[i] for i, v in enumerate(viol_mask) if v]
            if vt:
                ax.scatter(vt, vc, color="#ff4444", s=6, zorder=5,
                           label=f'violations ({len(vt)})', alpha=0.7)

        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Cumulative NN updates")
        ax.set_title("NN Updates vs Time")
        ax.legend(fontsize=8, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)
        txt = (f"Total updates : {nn.updates}\n"
               f"Violations    : {nn.violations}\n"
               f"Final LR      : {nn.lr:.2e}\n"
               f"Stable steps  : "
               f"{sum(1 for v in h['dVdt'] if v < 0)}/{len(h['dVdt'])}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top',
                fontsize=7.5, family='monospace', color="#88ccff",
                bbox=dict(boxstyle='round', fc=DARK, ec=SPINE, lw=1))

        # ── right: total (cumulative) loss vs update index ──
        ax = axs2[1]
        loss_arr = np.asarray(nn.loss_log) if nn.loss_log else np.array([])
        if len(loss_arr):
            # smooth with rolling mean (window=20) for readability
            win = min(20, len(loss_arr))
            loss_smooth = np.convolve(loss_arr,
                                      np.ones(win) / win, mode='valid')
            upd_idx     = np.arange(len(loss_arr))
            smooth_idx  = np.arange(len(loss_smooth)) + win // 2

            ax.fill_between(upd_idx, loss_arr,
                            alpha=0.18, color="#ff9944")
            ax.plot(upd_idx, loss_arr,
                    color="#ff9944", lw=0.8, alpha=0.5, label='raw loss')
            ax.plot(smooth_idx, loss_smooth,
                    color="#ffdd44", lw=2.2, label=f'smoothed (w={win})')

            # cumulative loss on twin axis
            ax2_twin = ax.twinx()
            ax2_twin.plot(upd_idx, np.cumsum(loss_arr),
                          color="#cc44ff", lw=1.5, ls='--', label='cumulative')
            ax2_twin.set_ylabel("Cumulative loss", color="#cc44ff", fontsize=8)
            ax2_twin.tick_params(colors="#cc44ff", labelsize=7)
            ax2_twin.spines['right'].set_color("#cc44ff")
            ax2_twin.set_facecolor("#060e1c")

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      fontsize=8, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        ax.set_xlabel("Update index")
        ax.set_ylabel("Step stability loss  ||oG||²")
        ax.set_title("Total Loss over Updates")

        fig2.tight_layout(rect=[0, 0, 1, 0.95])

        # ══════════════════════════════════════════════════════
        # FIGURE 3 — POSITION TRACKING
        # ══════════════════════════════════════════════════════
        fig3, axs3 = plt.subplots(2, 2, figsize=(13, 8), facecolor=DARK)
        fig3.suptitle("Position Tracking",
                      color=TC, fontsize=13, fontweight='bold', y=0.97)
        _style(fig3, axs3.flat)

        # ── top-left: X position ──
        ax = axs3[0, 0]
        ax.plot(t, h['x'],  color="#44ddff", lw=2.0, label='x actual')
        ax.plot(t, h['wx'], color="#ff8844", lw=1.5, ls='--', label='x waypoint')
        ax.axhline(self.GOAL[0], color="#ffd700", lw=1.0, ls=':', alpha=0.6,
                   label=f'goal x={self.GOAL[0]}')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("x (m)")
        ax.set_title("X Position vs Reference")
        ax.legend(fontsize=8, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        # ── top-right: Y position ──
        ax = axs3[0, 1]
        ax.plot(t, h['y'],  color="#44ff88", lw=2.0, label='y actual')
        ax.plot(t, h['wy'], color="#ff44aa", lw=1.5, ls='--', label='y waypoint')
        ax.axhline(self.GOAL[1], color="#ffd700", lw=1.0, ls=':', alpha=0.6,
                   label=f'goal y={self.GOAL[1]}')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("y (m)")
        ax.set_title("Y Position vs Reference")
        ax.legend(fontsize=8, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        # ── bottom-left: 2-D trajectory ──
        ax = axs3[1, 0]
        x_arr = np.asarray(h['x'])
        y_arr = np.asarray(h['y'])
        sc = ax.scatter(x_arr, y_arr,
                        c=np.linspace(0, 1, len(x_arr)),
                        cmap='cool', s=5, zorder=4)
        fig3.colorbar(sc, ax=ax).ax.tick_params(colors=MUTED, labelsize=7)
        ax.plot(*self.START, 'go', ms=10, label='start', zorder=6)
        ax.plot(*self.GOAL,  'y*', ms=14, label='goal',  zorder=6)
        for o in self.obstacles:
            ax.add_patch(Circle((o.x, o.y), o.r,
                                color='#aa2020', alpha=0.5, zorder=3))
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title("2-D Trajectory (colour = time)")
        ax.legend(fontsize=8, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        # ── bottom-right: position error norm ──
        ax = axs3[1, 1]
        pos_err = np.sqrt((x_arr - self.GOAL[0])**2 + (y_arr - self.GOAL[1])**2)
        ax.plot(t, pos_err, color="#ff6644", lw=2.0, label='dist to goal')
        ax.axhline(Controller.GOAL_TOL, color="#ffd700", lw=1.2, ls='--',
                   label=f'goal tol ({Controller.GOAL_TOL} m)')
        ax.fill_between(t, pos_err, alpha=0.15, color="#ff6644")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (m)")
        ax.set_title("Distance to Goal vs Time")
        ax.legend(fontsize=8, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        fig3.tight_layout(rect=[0, 0, 1, 0.95])

        # ══════════════════════════════════════════════════════
        # FIGURE 4 — VELOCITY TRACKING
        # ══════════════════════════════════════════════════════
        fig4, axs4 = plt.subplots(2, 2, figsize=(13, 8), facecolor=DARK)
        fig4.suptitle("Velocity Tracking",
                      color=TC, fontsize=13, fontweight='bold', y=0.97)
        _style(fig4, axs4.flat)

        spd   = np.asarray(h['spd'])
        v_des = np.asarray(h['v_des'])

        # ── top-left: actual vs desired speed ──
        ax = axs4[0, 0]
        ax.plot(t, spd,   color="#44ddff", lw=2.0, label='v actual')
        ax.plot(t, v_des, color="#ffdd44", lw=1.5, ls='--', label='v desired')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (m/s)")
        ax.set_title("Speed: Actual vs Desired")
        ax.legend(fontsize=8, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        # ── top-right: velocity error ──
        ax = axs4[0, 1]
        v_err = v_des - spd
        ax.plot(t, v_err, color="#ff6688", lw=1.8, label='v error = v_des - v')
        ax.axhline(0, color="#ffffff44", lw=0.8, ls='--')
        ax.fill_between(t, v_err, 0, where=(v_err > 0),
                        alpha=0.2, color="#ff4444", label='underspeed')
        ax.fill_between(t, v_err, 0, where=(v_err < 0),
                        alpha=0.2, color="#4444ff", label='overspeed')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("v error (m/s)")
        ax.set_title("Velocity Error")
        ax.legend(fontsize=8, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        # ── bottom-left: heading vs desired heading ──
        ax = axs4[1, 0]
        theta     = np.degrees(np.asarray(h['theta']))
        theta_des = np.degrees(np.asarray(h['theta_des']))
        ax.plot(t, theta,     color="#88ff44", lw=1.8, label='theta actual')
        ax.plot(t, theta_des, color="#ffaa22", lw=1.5, ls='--', label='theta desired')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Heading (deg)")
        ax.set_title("Heading: Actual vs Desired")
        ax.legend(fontsize=8, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        # ── bottom-right: NN velocity correction over time ──
        ax = axs4[1, 1]
        nn0 = np.asarray(h['nn0'])
        nn1 = np.asarray(h['nn1'])
        ax.plot(t, nn0, color="#cc44ff", lw=1.6, label='NN force corr')
        ax.plot(t, nn1, color="#44ffcc", lw=1.6, label='NN torque corr')
        ax.axhline(0, color="#ffffff33", lw=0.8, ls='--')
        ax.fill_between(t, nn0, alpha=0.12, color="#cc44ff")
        ax.fill_between(t, nn1, alpha=0.12, color="#44ffcc")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("NN output (normalised)")
        ax.set_title("NN Corrections (Adaptive Trim)")
        ax.legend(fontsize=8, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        fig4.tight_layout(rect=[0, 0, 1, 0.95])

        # ══════════════════════════════════════════════════════
        # FIGURE 5 — STABILITY ANALYSIS  (NEW)
        # ══════════════════════════════════════════════════════
        fig5, axs5 = plt.subplots(2, 3, figsize=(16, 9), facecolor=DARK)
        fig5.suptitle("Stability Analysis — Lyapunov & Adaptive SGD",
                      color=TC, fontsize=14, fontweight='bold', y=0.98)
        _style(fig5, axs5.flat)

        dVdt_arr  = np.asarray(h['dVdt'])
        V_arr     = np.asarray(h['V'])
        margin_arr= np.asarray(h['margin'])
        lr_arr    = np.asarray(h['lr'])
        enorm_arr = np.asarray(h['enorm'])

        # ── compute stability metrics ──────────────────────────
        stable_mask   = dVdt_arr < 0
        unstable_mask = dVdt_arr >= 0
        n_total    = len(dVdt_arr)
        n_stable   = int(np.sum(stable_mask))
        n_unstable = int(np.sum(unstable_mask))
        pct_stable = 100.0 * n_stable / max(n_total, 1)

        # stability gain = reduction in V from peak to final
        V_peak  = float(np.max(V_arr))   if len(V_arr) else 0.0
        V_final = float(V_arr[-1])       if len(V_arr) else 0.0
        V_init  = float(V_arr[0])        if len(V_arr) else 0.0
        stab_gain_pct = 100.0 * (V_init - V_final) / max(V_init, 1e-9)

        # mean stability margin (positive = inside stable region)
        mean_margin = float(np.mean(margin_arr)) if len(margin_arr) else 0.0

        # convergence time = first time V stays below 5% of initial
        conv_t = None
        thresh = 0.05 * V_init
        for idx in range(len(V_arr)):
            if np.all(V_arr[idx:] < thresh):
                conv_t = float(t[idx])
                break

        # ── plot 1: V(t) with stable/unstable shading ──────────
        ax = axs5[0, 0]
        ax.plot(t, V_arr, color="#c080ff", lw=2.0, label='V(e)', zorder=4)
        ax.fill_between(t, V_arr, where=stable_mask,
                        alpha=0.25, color="#44ff88", label='stable (dV/dt<0)')
        ax.fill_between(t, V_arr, where=unstable_mask,
                        alpha=0.35, color="#ff4444", label='unstable (dV/dt≥0)')
        if conv_t is not None:
            ax.axvline(conv_t, color="#ffd700", lw=1.5, ls='--',
                       label=f'converged t={conv_t:.1f}s')
        ax.axhline(thresh, color="#ffffff44", lw=0.9, ls=':',
                   label=f'5% V₀ threshold')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("V(e)")
        ax.set_title("Lyapunov V(e) — Stable vs Unstable Regions")
        ax.legend(fontsize=7, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        # ── plot 2: dV/dt with zero line ────────────────────────
        ax = axs5[0, 1]
        ax.plot(t, dVdt_arr, color="#ff9922", lw=1.6, label='dV/dt', zorder=4)
        ax.axhline(0, color="#ff443388", lw=1.2, ls='--', label='stability boundary')
        ax.fill_between(t, dVdt_arr, 0, where=(dVdt_arr < 0),
                        alpha=0.2, color="#44ff88", label='stable zone')
        ax.fill_between(t, dVdt_arr, 0, where=(dVdt_arr >= 0),
                        alpha=0.3, color="#ff4444", label='violation zone')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("dV/dt")
        ax.set_title("dV/dt — Lyapunov Derivative")
        ax.legend(fontsize=7, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        # ── plot 3: stability margin ────────────────────────────
        ax = axs5[0, 2]
        ax.plot(t, margin_arr, color="#44ffcc", lw=1.8, label='margin = -(dV/dt + 0.18V)')
        ax.axhline(0, color="#ffffff44", lw=1.0, ls='--')
        ax.fill_between(t, margin_arr, 0, where=(margin_arr > 0),
                        alpha=0.25, color="#44ffcc", label='positive margin (good)')
        ax.fill_between(t, margin_arr, 0, where=(margin_arr <= 0),
                        alpha=0.3,  color="#ff6644", label='negative margin (risk)')
        ax.axhline(mean_margin, color="#ffd700", lw=1.2, ls=':',
                   label=f'mean={mean_margin:.3f}')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Stability Margin")
        ax.set_title("Stability Margin  -(dV/dt + αV)")
        ax.legend(fontsize=7, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        # ── plot 4: adaptive LR vs dV/dt overlay ────────────────
        ax = axs5[1, 0]
        ax.plot(t, lr_arr, color="#4488ff", lw=2.0, label='Learning Rate', zorder=4)
        ax2b = ax.twinx()
        ax2b.plot(t, dVdt_arr, color="#ff992244", lw=1.0, label='dV/dt (ref)')
        ax2b.axhline(0, color="#ff443322", lw=0.8, ls='--')
        ax2b.set_ylabel("dV/dt", color="#ff992288", fontsize=8)
        ax2b.tick_params(colors="#ff9922", labelsize=7)
        ax2b.spines['right'].set_color("#ff9922")
        ax2b.set_facecolor(DARK)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Learning Rate")
        ax.set_title("Adaptive LR driven by dV/dt (SGD novelty)")
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2b.get_legend_handles_labels()
        ax.legend(lines1+lines2, labs1+labs2,
                  fontsize=7, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        # ── plot 5: error norm decay ─────────────────────────────
        ax = axs5[1, 1]
        ax.plot(t, enorm_arr, color="#ff6644", lw=1.8, label='‖e‖ error norm')
        ax.fill_between(t, enorm_arr, alpha=0.15, color="#ff6644")
        # exponential fit for decay rate annotation
        try:
            from numpy.polynomial import polynomial as Poly
            logE = np.log(np.clip(enorm_arr, 1e-6, None))
            coeffs = np.polyfit(t, logE, 1)
            decay_rate = coeffs[0]
            fit_line = np.exp(np.polyval(coeffs, t))
            ax.plot(t, fit_line, color="#ffd700", lw=1.5, ls='--',
                    label=f'exp fit λ={decay_rate:.3f}')
        except Exception:
            decay_rate = float('nan')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("‖e‖")
        ax.set_title("Error Norm Decay")
        ax.legend(fontsize=7, labelcolor=MUTED, facecolor=DARK, edgecolor=SPINE)

        # ── plot 6: stability gained — bar summary ───────────────
        ax = axs5[1, 2]
        ax.set_facecolor("#060e1c")
        categories  = ['Stable\nSteps %', 'V Reduction\n%', 'Mean\nMargin×100']
        values      = [pct_stable, max(stab_gain_pct, 0), mean_margin * 100]
        colors_bar  = ["#44ff88", "#c080ff", "#44ffcc"]
        bars = ax.bar(categories, values, color=colors_bar, width=0.5,
                      edgecolor=SPINE, linewidth=1.2, zorder=4)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(values)*0.02,
                    f'{val:.1f}', ha='center', va='bottom',
                    color="#ffffff", fontsize=10, fontweight='bold')
        ax.set_ylabel("Value"); ax.set_title("Stability Gained — Summary")
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
        ax.axhline(100, color="#ffffff22", lw=0.8, ls=':')

        fig5.tight_layout(rect=[0, 0, 1, 0.95])

        # ══════════════════════════════════════════════════════
        # POPUP — HOW MUCH STABILITY WAS GAINED  (NEW)
        # ══════════════════════════════════════════════════════
        popup = plt.figure(figsize=(6.5, 5.2), facecolor="#07111f")
        popup.suptitle("⚡  STABILITY GAINED  ⚡",
                       color="#ffd700", fontsize=16, fontweight='bold', y=0.97)
        ax_pop = popup.add_subplot(111)
        ax_pop.set_facecolor("#07111f")
        ax_pop.axis('off')

        # decide overall stability rating
        if pct_stable >= 85:
            rating = "EXCELLENT ✅"
            rating_color = "#44ff88"
        elif pct_stable >= 65:
            rating = "GOOD  ✔"
            rating_color = "#aaff44"
        elif pct_stable >= 45:
            rating = "MODERATE ⚠"
            rating_color = "#ffdd44"
        else:
            rating = "POOR ✖"
            rating_color = "#ff4444"

        conv_str = f"{conv_t:.2f} s" if conv_t is not None else "not reached"

        summary_lines = [
            ("Overall Stability Rating",  rating,               rating_color),
            ("",                          "",                    "#ffffff"),
            ("Stable timesteps",          f"{n_stable} / {n_total}  ({pct_stable:.1f}%)",  "#44ff88"),
            ("Unstable timesteps",        f"{n_unstable}  ({100-pct_stable:.1f}%)",         "#ff6644"),
            ("",                          "",                    "#ffffff"),
            ("Initial  V(e₀)",            f"{V_init:.4f}",      "#c080ff"),
            ("Final    V(e_f)",            f"{V_final:.4f}",     "#c080ff"),
            ("V Reduction (stability gain)", f"{stab_gain_pct:.1f} %",  "#44ffcc"),
            ("Peak V encountered",        f"{V_peak:.4f}",      "#ff9922"),
            ("",                          "",                    "#ffffff"),
            ("Convergence time",          conv_str,              "#ffd700"),
            ("Mean stability margin",     f"{mean_margin:.4f}",  "#44ffcc"),
            ("LR violations triggered",  f"{nn.violations}",    "#ff88aa"),
            ("Final learning rate",       f"{nn.lr:.2e}",        "#4488ff"),
            ("Total NN updates",          f"{nn.updates}",       "#88aaff"),
        ]

        y_pos = 0.95
        for label, value, color in summary_lines:
            if label == "":
                y_pos -= 0.025
                continue
            if label == "Overall Stability Rating":
                ax_pop.text(0.05, y_pos, label + ":",
                            transform=ax_pop.transAxes,
                            fontsize=11, color="#aaccdd",
                            fontweight='bold', family='monospace')
                ax_pop.text(0.62, y_pos, value,
                            transform=ax_pop.transAxes,
                            fontsize=12, color=color,
                            fontweight='bold', family='monospace')
            else:
                ax_pop.text(0.05, y_pos, label + ":",
                            transform=ax_pop.transAxes,
                            fontsize=9, color="#6090b8", family='monospace')
                ax_pop.text(0.62, y_pos, value,
                            transform=ax_pop.transAxes,
                            fontsize=9, color=color,
                            fontweight='bold', family='monospace')
            y_pos -= 0.058

        # draw a coloured border rect around rating
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                              boxstyle="round,pad=0.02",
                              linewidth=2.5, edgecolor=rating_color,
                              facecolor="none",
                              transform=ax_pop.transAxes, zorder=10)
        ax_pop.add_patch(rect)

        popup.tight_layout(rect=[0, 0, 1, 0.94])

        print("\n[POST-SIM] Showing 5 analysis figures + stability popup — close windows to exit.")
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Adaptive Neural Lyapunov Control")
    parser.add_argument('--load',    type=str, default="mlp_weights.npz")
    parser.add_argument('--collect', type=str)
    parser.add_argument('--auto',    action='store_true')
    parser.add_argument('--play',    action='store_true')

    if len(sys.argv) == 1:
        args = parser.parse_args(['--play'])
    else:
        args = parser.parse_args()

    banner = "=" * 68
    print(banner)
    print("  ADAPTIVE NEURAL LYAPUNOV - STABILITY LEARNING")
    print("  MLP 8->16->12->2  -  Stability-Aware SGD  -  A*")
    print(banner)
    print()
    print("CONTROLS (PLACE phase):")
    print("  Left-click  add obstacle     Right-click  remove obstacle")
    print("  SPACE       start            R            reset")
    print("  1           A*")
    print("  + / -       obstacle radius  P  force replan")
    print()
    print("CONTROLS (RUN phase)  <-- NEW:")
    print("  Left-click  place LIVE obstacle + instant replan")
    print("  Right-click remove obstacle     + instant replan")
    print()
    print("ALGORITHMS:")
    print("  1  A*       optimal grid shortest path")
    print()
    print(f"Backend: {matplotlib.get_backend()}")
    print(banner)

    sim = Simulation(collect_path=args.collect, auto_start=args.auto)

    if args.load and os.path.exists(args.load):
        print(f"Weights will be loaded from {args.load} when simulation starts.")
        original_init = Controller.__init__
        def new_init(self, *init_args, **kwargs):
            original_init(self, *init_args, **kwargs)
        Controller.__init__ = new_init

    try:
        sim.run()
    except KeyboardInterrupt:
        pass
    finally:
        sim.save_collected_data()
        sim.save_log_csv()
        sim.plot_results()

    print("Simulation finished.")


if __name__ == "__main__":
    main()
