import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

def animate_2R_planar_arm_traj(
    joint_xs: NDArray[np.double], joint_ys: NDArray[np.double], animation_file_name: str
) -> tuple[plt.Figure, plt.Axes, animation.FuncAnimation]:
    fig = plt.figure(figsize=(5, 5))
    limit = 2.3
    ax = fig.add_subplot(autoscale_on=False, xlim=(-limit, limit), ylim=(-limit, limit))
    ax.set_aspect("equal")

    fourth_quadrant_obstacle = Rectangle(
        xy=(0, -limit),
        width=limit,
        height=limit,
        facecolor="black",
        linewidth=0,
        zorder=0,
    )
    ax.add_patch(fourth_quadrant_obstacle)
    ceiling_obstacle = Rectangle(
        xy=(-limit, 1.5),
        width=2 * limit,
        height=limit - 1.5,
        facecolor="black",
        linewidth=0,
        zorder=0,
    )
    ax.add_patch(ceiling_obstacle)

    (link1,) = ax.plot([], [], "o-", color="blue", lw=2)
    (link2,) = ax.plot([], [], "o-", color="red", lw=2)
    time_template = "t = %.1fs"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def animate(t):
        xs1 = [joint_xs[0, t], joint_xs[1, t]]
        xs2 = [joint_xs[1, t], joint_xs[2, t]]
        ys1 = [joint_ys[0, t], joint_ys[1, t]]
        ys2 = [joint_ys[1, t], joint_ys[2, t]]

        link1.set_data(xs1, ys1)
        link2.set_data(xs2, ys2)
        time_text.set_text(time_template % (t * 1e-3))
        return link1, link2, time_text

    ani = animation.FuncAnimation(
        fig, animate, interval=1, blit=True, save_count=300
    )

    # gif_writer = animation.PillowWriter(fps=30)
    # ani.save(f"{animation_file_name}.gif", writer=gif_writer)
    # print(f"Saved animation to '{animation_file_name}.gif'")
    return fig, ax, ani


def plot_snapshots(
    dt: float,
    joint_xs: NDArray[np.double],
    joint_ys: NDArray[np.double],
    joint_xs_desired: NDArray[np.double] | None = None,
    joint_ys_desired: NDArray[np.double] | None = None,
    control_period_s: float = 1e-3
) -> tuple[plt.Figure, plt.Axes]:
    step_size = round(dt // control_period_s)
    ixs = np.arange(start=0, stop=joint_xs.shape[1], step=step_size)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    limit = 2.3
    fourth_quadrant_obstacle = Rectangle(
        xy=(0, -limit),
        width=limit,
        height=limit,
        facecolor="black",
        linewidth=0,
        zorder=0,
    )
    ax.add_patch(fourth_quadrant_obstacle)
    ceiling_obstacle = Rectangle(
        xy=(-limit, 1.5),
        width=2 * limit,
        height=limit - 1.5,
        facecolor="black",
        linewidth=0,
        zorder=0,
    )

    def opacity_function(fraction: float, min_alpha: float = 0.4) -> float:
        fraction = max(0.0, min(fraction, 1.0))
        return min(1.0, max(fraction ** 3, min_alpha))

    ax.add_patch(ceiling_obstacle)
    for i, ix in enumerate(ixs):
        if joint_xs_desired is not None and joint_ys_desired is not None:
            x0, y0 = joint_xs_desired[0, ix], joint_ys_desired[0, ix]
            x1, y1 = joint_xs_desired[1, ix], joint_ys_desired[1, ix]
            x2, y2 = joint_xs_desired[2, ix], joint_ys_desired[2, ix]
            ax.plot(
                [x0, x1],
                [y0, y1],
                "o--",
                color="blue",
                lw=2,
                markersize=4,
                alpha=opacity_function(i / len(ixs)),
            )
            ax.plot(
                [x1, x2],
                [y1, y2],
                "o--",
                color="red",
                lw=2,
                markersize=4,
                alpha=opacity_function(i / len(ixs)),
            )

        x0, y0 = joint_xs[0, ix], joint_ys[0, ix]
        x1, y1 = joint_xs[1, ix], joint_ys[1, ix]
        x2, y2 = joint_xs[2, ix], joint_ys[2, ix]
        ax.plot(
            [x0, x1],
            [y0, y1],
            "o-",
            color="blue",
            lw=2,
            markersize=4,
            alpha=opacity_function(i / len(ixs)),
        )
        ax.plot(
            [x1, x2],
            [y1, y2],
            "o-",
            color="red",
            lw=2,
            markersize=4,
            alpha=opacity_function(i / len(ixs)),
        )

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")

    return fig, ax
