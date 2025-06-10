import numpy as np
from numpy.typing import NDArray
from pydrake.systems.framework import LeafSystem, BasicVector, Context
from scipy.interpolate import CubicSpline


class JointSpaceTrajectorySource(LeafSystem):
    def __init__(
        self,
        name: str,
        num_joints: int,
        times: NDArray[np.double],
        joint_coordinates: NDArray[np.double],
    ):
        super().__init__()

        self.num_joints = max(round(abs(num_joints)), 1)
        self.times = np.asarray(times, dtype=np.double)
        self.joint_coordinates = np.asarray(joint_coordinates, dtype=np.double)

        self.ix: int = 0
        self.DeclareVectorOutputPort(
            name=name, size=self.num_joints, calc=self.calc_output
        )

    def calc_output(self, context: Context, output: BasicVector):
        t = context.get_time()

        if t >= self.times[self.ix]:
            self.ix += 1

        self.ix = min(self.ix, len(self.times) - 1)

        output.set_value(self.joint_coordinates[:, self.ix])


def eval_cubic_spline_traj(
    times: NDArray[np.double],
    waypoint_times: NDArray[np.double],
    waypoints: NDArray[np.double],
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
    times = np.asarray(times, dtype=np.double)
    waypoint_times = np.asarray(waypoint_times, dtype=np.double)
    waypoints = np.asarray(waypoints, dtype=np.double)

    spl = CubicSpline(x=waypoint_times, y=waypoints, axis=1, bc_type="clamped")

    return spl(times), spl(times, 1), spl(times, 2)

