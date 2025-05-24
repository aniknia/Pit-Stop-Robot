from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def calc_fk_2D(
    link_lens: Sequence[float], joint_positions: NDArray[np.double]
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """
    Calculates the x and y positions of all three frames of a planar 2R manipulator
    (including end effector and base joint) given manipulator link lengths and a
    configuration

    Parameters
    ----------
    link_lens
        A NumPy array of shape (2,) containing the three link lengths of the planar 2R
        manipulator (ordered from link 1 to link 2)

    joint_positions
        A NumPy array of shape (2, N) containing the joint positions of the planar
        2R manipulator (ordered from theta1 to theta2 along the first axis)

    Returns
    -------
    A tuple of two NumPy array both with shape (3, N). The first and second element of
    the tuple contain the x coordinates and y coordinates for the origin of each frame
    (joints and end-effector) of the planar 2R manipulator, respectively.
    """
    L1, L2 = link_lens
    x1 = L1 * np.cos(joint_positions[0, :])
    y1 = L1 * np.sin(joint_positions[0, :])

    x2 = x1 + L2 * np.cos(joint_positions[0, :] + joint_positions[1, :])
    y2 = y1 + L2 * np.sin(joint_positions[0, :] + joint_positions[1, :])

    joint_xs = np.stack([np.zeros_like(x1), x1, x2], axis=1).T
    joint_ys = np.stack([np.zeros_like(y1), y1, y2], axis=1).T

    return joint_xs, joint_ys
