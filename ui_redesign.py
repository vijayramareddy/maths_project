import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation
import numpy as np


class ModernDashboard:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 9), facecolor="#07131d")
        self.gs = self.fig.add_gridspec(2, 3, left=0.04, right=0.96, top=0.9, bottom=0.08,
                                       wspace=0.22, hspace=0.26)

        self.ax_main = self.fig.add_subplot(self.gs[:, 0:2])
        self.ax_status = self.fig.add_subplot(self.gs[0, 2])
        self.ax_lr = self.fig.add_subplot(self.gs[1, 2])

        self._theme()
        self._build_main_scene()
        self._build_status_cards()
        self._build_lr_panel()

    def _theme(self):
        bg = "#07131d"
        panel = "#0d1b2a"
        grid = "#173553"
        text = "#dfeefe"
        muted = "#8aa9c3"
        accent = "#72f1d7"
        alert = "#ffd166"
        danger = "#ff6b6b"

        self.bg = bg
        self.panel = panel
        self.grid = grid
        self.text = text
        self.muted = muted
        self.accent = accent
        self.alert = alert
        self.danger = danger

        for ax in (self.ax_main, self.ax_status, self.ax_lr):
            ax.set_facecolor(panel)
            for spine in ax.spines.values():
                spine.set_color("#23486c")
            ax.tick_params(colors=muted, labelsize=8)
            ax.grid(True, color=grid, linewidth=0.8, alpha=0.8)

    def _build_main_scene(self):
        self.ax_main.set_xlim(0, 10)
        self.ax_main.set_ylim(0, 10)
        self.ax_main.set_aspect("equal")
        self.ax_main.set_title("Adaptive Neural Lyapunov Control", color=self.accent, fontsize=15, weight="bold", pad=12)

        self.ax_main.set_xlabel("X (m)", color=self.muted)
        self.ax_main.set_ylabel("Y (m)", color=self.muted)

        # Arena decorations
        start = (1.3, 1.3)
        goal = (8.7, 8.5)

        self.ax_main.text(0.02, 0.96, "LIVE PLANNER", transform=self.ax_main.transAxes,
                          color=self.text, fontsize=9, weight="bold",
                          bbox=dict(boxstyle="round,pad=0.35", facecolor="#0a1727", edgecolor=self.accent, linewidth=1.2))

        self.goal_circle = Circle(goal, 0.28, color="#ffd166", ec="#b98600", lw=2, zorder=5)
        self.ax_main.add_patch(self.goal_circle)
        self.ax_main.text(*goal, "G", ha="center", va="center", fontsize=10, weight="bold", color="#111111")

        self.start_circle = Circle(start, 0.26, color="#72f1d7", ec="#1c8c7d", lw=2, zorder=5)
        self.ax_main.add_patch(self.start_circle)
        self.ax_main.text(*start, "S", ha="center", va="center", fontsize=10, weight="bold", color="#111111")

        for i, (cx, cy, r) in enumerate([(2.3, 3.0, 0.65), (4.8, 5.0, 0.7), (6.3, 3.8, 0.6), (7.5, 6.7, 0.55)]):
            patch = Circle((cx, cy), r, color="#ff7a7a", ec="#ffb8b8", lw=1.5, alpha=0.85, zorder=4)
            self.ax_main.add_patch(patch)

        self.path_line, = self.ax_main.plot([], [], color=self.accent, lw=2.6, alpha=0.9, zorder=3)
        self.trajectory, = self.ax_main.plot([], [], color="#7db8ff", lw=1.8, alpha=0.95, zorder=2)
        self.robot = FancyArrowPatch((1.3, 1.3), (1.3 + 0.45, 1.3 + 0.0), arrowstyle='-|>', mutation_scale=18,
                                    color="#7db8ff", linewidth=2.2, zorder=6)
        self.ax_main.add_patch(self.robot)

    def _build_status_cards(self):
        self.ax_status.set_title("SYSTEM STATUS", color=self.text, fontsize=10, pad=8)
        self.ax_status.text(0.06, 0.82, "Mode", transform=self.ax_status.transAxes,
                            color=self.muted, fontsize=8)
        self.ax_status.text(0.55, 0.82, "RUNNING", transform=self.ax_status.transAxes,
                            color=self.accent, fontsize=9, weight="bold")

        self.ax_status.text(0.06, 0.66, "Goal distance", transform=self.ax_status.transAxes,
                            color=self.muted, fontsize=8)
        self.ax_status.text(0.55, 0.66, "0.42 m", transform=self.ax_status.transAxes,
                            color=self.alert, fontsize=9, weight="bold")

        self.ax_status.text(0.06, 0.50, "Replans", transform=self.ax_status.transAxes,
                            color=self.muted, fontsize=8)
        self.ax_status.text(0.55, 0.50, "03", transform=self.ax_status.transAxes,
                            color=self.accent, fontsize=9, weight="bold")

        self.ax_status.text(0.06, 0.34, "Stability", transform=self.ax_status.transAxes,
                            color=self.muted, fontsize=8)
        self.ax_status.text(0.55, 0.34, "94%", transform=self.ax_status.transAxes,
                            color="#8cffac", fontsize=9, weight="bold")

        self.ax_status.text(0.06, 0.18, "Controller", transform=self.ax_status.transAxes,
                            color=self.muted, fontsize=8)
        self.ax_status.text(0.55, 0.18, "A* + NN", transform=self.ax_status.transAxes,
                            color=self.text, fontsize=9, weight="bold")

        self.ax_status.set_xticks([])
        self.ax_status.set_yticks([])

    def _build_lr_panel(self):
        self.ax_lr.set_title("LIVE LEARNING RATE", color=self.text, fontsize=10, pad=8)
        t = np.linspace(0, 1, 80)
        y = 0.004 + 0.003 * np.sin(6 * t * np.pi) + 0.002 * np.exp(-(t - 0.6) ** 2 / 0.08)
        self.lr_line, = self.ax_lr.plot(t, y, color=self.accent, lw=2.5)
        self.ax_lr.set_ylim(0.0, 0.012)
        self.ax_lr.set_xticks([])
        self.ax_lr.set_yticks([])

    def set_path(self, waypoints):
        xs = [p[0] for p in waypoints]
        ys = [p[1] for p in waypoints]
        self.path_line.set_data(xs, ys)
        self.trajectory.set_data([1.3], [1.3])

    def animate(self, i):
        theta = i * 0.08
        x = 1.3 + 0.7 * math.sin(theta) + 0.13 * i
        y = 1.3 + 0.7 * math.cos(theta) + 0.1 * i
        x = min(max(x, 0.5), 9.4)
        y = min(max(y, 0.5), 9.4)

        dx = 0.5 * math.cos(theta)
        dy = 0.5 * math.sin(theta)
        self.robot.remove()
        self.robot = FancyArrowPatch((x, y), (x + dx, y + dy), arrowstyle='-|>', mutation_scale=18,
                                    color="#7db8ff", linewidth=2.2, zorder=6)
        self.ax_main.add_patch(self.robot)

        self.trajectory.set_data(np.append(self.trajectory.get_xdata(), x), np.append(self.trajectory.get_ydata(), y))
        if len(self.trajectory.get_xdata()) > 45:
            self.trajectory.set_data(self.trajectory.get_xdata()[-45:], self.trajectory.get_ydata()[-45:])

        return self.robot, self.trajectory, self.path_line

    def run(self):
        anim = FuncAnimation(self.fig, self.animate, frames=120, interval=35, blit=False)
        plt.show()


if __name__ == "__main__":
    demo = ModernDashboard()
    demo.set_path([(1.3, 1.3), (2.0, 2.1), (3.2, 3.6), (4.6, 4.2), (5.9, 5.4), (7.3, 7.1), (8.7, 8.5)])
    demo.run()
