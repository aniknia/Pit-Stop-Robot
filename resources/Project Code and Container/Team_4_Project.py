"""
IMPORTANT NOTE:
    The instructions for completing this template are inline with the code. You can
    find them by searching for: "TODO:"
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, Diagram, Context
from pydrake.systems.primitives import MatrixGain, LogVectorOutput

from mechae263C_helpers.drake import LinearCombination, plot_diagram
from mechae263C_helpers.hw4.arm import Arm
from mechae263C_helpers.hw4.kinematics import calc_fk_2D
from mechae263C_helpers.hw4.trajectory import (
    eval_cubic_spline_traj,
    JointSpaceTrajectorySource,
)
from mechae263C_helpers.hw4.plotting import animate_2R_planar_arm_traj, plot_snapshots
import matplotlib.animation as animation


def run_simulation(
    q_initial: NDArray[np.double],
    q_final: NDArray[np.double],
    B_avg: NDArray[np.double],
    K_p: NDArray[np.double],
    K_d: NDArray[np.double],
    simulation_duration_s: float,
    should_apply_control_torques: bool,
    control_period_s: float = 1e-3,
) -> tuple[
    NDArray[np.double],
    tuple[NDArray[np.double], NDArray[np.double]],
    tuple[NDArray[np.double], NDArray[np.double]],
    NDArray[np.double],
    Diagram,
]:
    """
    Runs a simulation with a desired joint position

    Parameters
    ----------
    q_initial:
        A numpy array of shape (2,) containing the initial joint positions

    q_final:
        A numpy array of shape (2,) containing the final desired joint positions

    B_avg:
        A numpy array of shape (2, 2) containing the average linearized inertia matrix

    K_p:
        A numpy array of shape (2, 2) containing the proportional gains of the inverse
        dynamics controller.

    K_d:
        A numpy array of shape (2, 2) containing the derivative gains of the inverse
        dynamics controller.

    control_period_s:
        The period between control commands in units of seconds

    simulation_duration_s:
        The duration of the simulation in units of seconds

    should_apply_control_torques:
        A bool that specifies that control torques should be simulated when set to
        `True`. (If set to `False` then no control torques are simulated).

    Returns
    -------
    A tuple with five elements:
        1. A numpy array with shape (T,) of simulation time steps
        2. A tuple of numpy arrays both with shape (2, T) of desired and actual joint
           positions corresponding to each simulation time step, respectively.
        3. A tuple of numpy arrays both with shape (2, T) of desired and actual joint
           velocities corresponding to each simulation time step, respectively.
        4. A numpy array with shape (2, T) of applied control torques corresponding to
           each simulation time step
        5. A Drake diagram
    """
    # ----------------------------------------------------------------------------------
    # Add "systems" to a `DiagramBuilder` object.
    #   - "systems" are the blocks in a block diagram
    #   - Some examples for how to add named systems to a `DiagramBuilder` are given
    #     below
    # ----------------------------------------------------------------------------------
    builder = DiagramBuilder()

    # Create the desired joint angle, velocity, and acceleration trajectories
    dt = control_period_s
    times = np.arange(0, simulation_duration_s + dt, dt)
    waypoint_times = np.asarray([0, simulation_duration_s / 2, simulation_duration_s])
    waypoints = np.stack([q_initial, np.deg2rad([130, -110]), q_final], axis=1)

    q_d_traj, qdot_d_traj, qddot_d_traj = eval_cubic_spline_traj(
        times=times, waypoint_times=waypoint_times, waypoints=waypoints
    )
    q_traj = builder.AddNamedSystem(
        "q_d_traj",
        JointSpaceTrajectorySource(
            name="q_d_traj",
            num_joints=q_d_traj.shape[0],
            times=times,
            joint_coordinates=q_d_traj,
        ),
    )
    qdot_traj = builder.AddNamedSystem(
        "qdot_d_traj",
        JointSpaceTrajectorySource(
            name="qdot_d_traj",
            num_joints=qdot_d_traj.shape[0],
            times=times,
            joint_coordinates=qdot_d_traj,
        ),
    )
    if should_apply_control_torques:
        qddot_traj = builder.AddNamedSystem(
            "qddot_d_traj",
            JointSpaceTrajectorySource(
                name="qddot_d_traj",
                num_joints=qddot_d_traj.shape[0],
                times=times,
                joint_coordinates=qddot_d_traj,
            ),
        )

        K_p_gain = builder.AddNamedSystem(
            "K_p", MatrixGain(np.asarray(K_p, dtype=np.double))
        )
        K_d_gain = builder.AddNamedSystem(
            "K_d", MatrixGain(np.asarray(K_d, dtype=np.double))
        )

    joint_position_error = builder.AddNamedSystem(
        "joint_position_error",
        LinearCombination(input_coeffs=(1, -1), input_shapes=(2,)),
    )
    joint_velocity_error = builder.AddNamedSystem(
        "joint_velocity_error",
        LinearCombination(input_coeffs=(1, -1), input_shapes=(2,)),
    )
    arm = builder.AddNamedSystem("arm", Arm())

    if should_apply_control_torques:
        control_torque = builder.AddNamedSystem(
            "u", LinearCombination(input_coeffs=(1, 1, 1), input_shapes=(2,))
        )
        inertia_matrix = builder.AddNamedSystem("B_avg", MatrixGain(B_avg))

    # ----------------------------------------------------------------------------------
    # Connect the systems in the `DiagramBuilder` (i.e. add arrows of block diagram)
    # ----------------------------------------------------------------------------------
    # `builder.ExportInput(input_port)` makes the provided "input_port" into an input
    # of the entire diagram
    # The functions system.get_input_port() returns the input port of the given system
    #   - If there is more than one input port, you must specify the index of the
    #     desired input
    # The functions system.get_output_port() returns the output port of the given system
    #   - If there is more than one output port, you must specify the index of the
    #     desired output
    builder.Connect(q_traj.get_output_port(), joint_position_error.get_input_port(0))
    builder.Connect(qdot_traj.get_output_port(), joint_velocity_error.get_input_port(0))
    if should_apply_control_torques:
        builder.Connect(qddot_traj.get_output_port(), inertia_matrix.get_input_port())

    joint_velocity_output = arm.get_output_port(0)
    joint_position_output = arm.get_output_port(1)

    # TODO:
    #   Replace any `...` below with the correct system and values. Please keep the
    #   system names the same
    builder.Connect(joint_position_output, joint_position_error.get_input_port(1))
    builder.Connect(joint_velocity_output, joint_velocity_error.get_input_port(1))
    if should_apply_control_torques:
        builder.Connect(joint_position_error.get_output_port(), K_p_gain.get_input_port())
        builder.Connect(joint_velocity_error.get_output_port(), K_d_gain.get_input_port())

        #
        builder.Connect(
            inertia_matrix.get_output_port(), control_torque.get_input_port(0)
        )
        builder.Connect(K_p_gain.get_output_port(), control_torque.get_input_port(1))
        builder.Connect(K_d_gain.get_output_port(), control_torque.get_input_port(2))
        builder.Connect(control_torque.get_output_port(), arm.get_input_port())
    else:
        builder.ExportInput(arm.get_input_port(), name="control_torque")

    # ----------------------------------------------------------------------------------
    # Log joint positions
    # ----------------------------------------------------------------------------------
    # These systems are special in Drake. They periodically save the output port value
    # a during a simulation so that it can be accessed later. The value is saved every
    # `publish_period` seconds in simulation time.
    joint_position_logger = LogVectorOutput(
        arm.get_output_port(1), builder, publish_period=control_period_s
    )
    joint_velocity_logger = LogVectorOutput(
        arm.get_output_port(0), builder, publish_period=control_period_s
    )
    if should_apply_control_torques:
        control_torque_logger = LogVectorOutput(
            control_torque.get_output_port(), builder, publish_period=control_period_s
        )

    # ----------------------------------------------------------------------------------
    # Setup/Run the simulation
    # ----------------------------------------------------------------------------------
    # This line builds a `Diagram` object and uses it to make a `Simulator` object for
    # the diagram
    diagram: Diagram = builder.Build()
    diagram.set_name("Inverse Dynamics Controller")
    simulator: Simulator = Simulator(diagram)

    # Get the context (this contains all the information needed to run the simulation)
    context: Context = simulator.get_mutable_context()

    # Set initial conditions
    initial_conditions = context.get_mutable_continuous_state_vector()
    initial_conditions.SetAtIndex(2, q_initial[0])
    initial_conditions.SetAtIndex(3, q_initial[1])

    if not should_apply_control_torques:
        diagram.get_input_port().FixValue(context, np.zeros((2,)))

    # Advance the simulation by `simulation_duration_s` seconds using the
    # `simulator.AdvanceTo()` function
    simulator.AdvanceTo(simulation_duration_s)

    # ----------------------------------------------------------------------------------
    # Extract simulation outputs
    # ----------------------------------------------------------------------------------
    # The lines below extract the joint position log from the simulator context
    joint_position_log = joint_position_logger.FindLog(simulator.get_context())
    t = joint_position_log.sample_times()
    q_actual = joint_position_log.data()

    joint_velocity_log = joint_velocity_logger.FindLog(simulator.get_context())
    qdot_actual = joint_velocity_log.data()

    control_torques = np.zeros((2, len(t)), dtype=np.double)

    if should_apply_control_torques:
        control_torque_log = control_torque_logger.FindLog(simulator.get_context())
        control_torques = control_torque_log.data()

    # Return a `tuple` of required results
    return t, (q_d_traj, q_actual), (qdot_d_traj, qdot_actual), control_torques, diagram


if __name__ == "__main__":
    ####################################################################################
    # Section 1
    ####################################################################################
    # ----------------------------------------------------------------------------------
    # TODO:
    #   Replace `...` with the correct values for each parameter
    # ----------------------------------------------------------------------------------
    # The below functions might be helpful:
    #   np.diag: https://numpy.org/doc/stable/reference/generated/numpy.diag.html
    #   np.eye: https://numpy.org/doc/stable/reference/generated/numpy.eye.html
    #link_lens = np.asarray([250, 150, 60]), link_lens = np.asarray([150, 100, 60])

    #Note: replace a1, a2 etc values with link_lens
    #Note: replace l1, l2 etc values with link_lens/2 for center of masses
    link_lens_long = np.array([.250, .150, .60]) #meters
    link_lens_medium = np.asarray([.200, .125, .60])
    link_lens_short = np.asarray([.150, .100, .60])
    link_lens_ex_short = np.asarray([.130, .90, .60])

    #Note: need to figure out mass oand moments of inertia for arms and motors
    m_l1 = m_l2 = 9
    I_l1 = I_l2 = 3

    #values from documentation, assuming all motors are the same
    m_m = 0.077 #kg
    I_m = 1/12*m_m1*(0.0356**2 + 0.00506**2) #kg*m^2, assuming the motors are rectangular prisms
    #gear ratio listed as 193:1 in spec sheet
    k_r = 193

    K_p = np.diag([4000, 4000])
    K_d = np.diag([1500, 1500])
    print("\tK_p Matrix:\n", K_p)
    print("\tK_d Matrix:\n", K_d)


    # ----------------------------------------------------------------------------------
    # TODO:
    #   Replace `...` with the correct values for the diagonal terms of the "averaged"
    #   generalized inertia matrix. These terms can be found by taking the small angle
    #   approximation of the elements in the full generalized inertia matrix given in
    #   the problem statement.
    # ----------------------------------------------------------------------------------
    B_avg = np.zeros((3, 3))
    B_avg[0, 0] = 
    B_avg[1, 1] = I_l1 + m_l1*l_1**2 + k_r1**2*I_m1 + I_l2 + m_l2*(a_1**2 + l_2**2 + 2*a_1*l_2*1) #using cos(q)=1 for small angle approx
    B_avg[2, 2] = I_l2 + m_l2*l_2**2 + k_r2**2*I_m2

    # ----------------------------------------------------------------------------------
    # TODO:
    #   Replace `...` with the initial and final joint configurations specified in the
    #   problem statement.
    # ----------------------------------------------------------------------------------
    q_initial = np.transpose(np.deg2rad([30, -60]))
    q_final = np.transpose(np.deg2rad([240, 60]))

    # ----------------------------------------------------------------------------------
    # Simulate without control torques
    # TODO:
    #   Replace `...` with the correct values to simulate the un-actuated dynamics of
    #   the planar 2R manipulator.
    # ----------------------------------------------------------------------------------
    t, (q_d, q), (_, _), _, diagram = run_simulation(
        q_initial=q_initial,
        q_final=q_final,
        B_avg=B_avg,
        K_p=K_p,
        K_d=K_d,
        simulation_duration_s=2.5,
        should_apply_control_torques=False,  # True or False?
    )
    # Convert `q` and `q_d` to degrees
    q_d = np.rad2deg(q_d)
    q = np.rad2deg(q)
    fig, ax = plot_diagram(diagram)
    fig.savefig('Block_Diagram_Without_Control_Torques.png', dpi=300)

    # ----------------------------------------------------------------------------------
    # TODO:
    #   Using the link lengths `[a_1, a_2]`, the simulated joint positions `q`, and the
    #   `calc_fk_2D` function to calculate the xy positions of each joint of the
    #   manipulator for the simulated scenario. (Replace `...` with the correct values)
    #
    #   Hint: Make sure to convert `q` back to radians before using it with `calc_fk_2D`
    #         (using np.deg2rad).
    #
    # ----------------------------------------------------------------------------------
    joint_xs, joint_ys = calc_fk_2D(link_lens=[a_1, a_2], joint_positions=np.deg2rad(q))

    # ----------------------------------------------------------------------------------
    # TODO:
    #   Replace all `...` in the call of the `animate_2R_planar_arm_traj` function with
    #   the correct output of the `calc_fk_2D`.
    # ----------------------------------------------------------------------------------
    _, _, anim_no_control_torques = animate_2R_planar_arm_traj(
        joint_xs=joint_xs, 
        joint_ys=joint_ys,        
        animation_file_name="no_control_torques_animation"
    )

    # ----------------------------------------------------------------------------------
    # Plot Snapshots
    # TODO:
    #   Replace all `...` in the call of the `plot_snapshots` function with
    #   the correct output of the `calc_fk_2D` and the dt specified in the problem
    #   statement.
    #   Add code to properly label `ax` and save `fig`
    # ----------------------------------------------------------------------------------
    fig, ax = plot_snapshots(dt=0.1, joint_xs=joint_xs, joint_ys=joint_ys)
    ax.set_title("Joint Animation Without Control Torques")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.savefig("without_control_torques.png", dpi=300)


    ####################################################################################
    # Section 2
    ####################################################################################
    # ----------------------------------------------------------------------------------
    # Simulate with control torques
    # TODO:
    #   Replace `...` with the correct values to simulate the dynamics of
    #   the planar 2R manipulator under your inverse dynamics controller.
    # ----------------------------------------------------------------------------------
    t, (q_d, q), (qdot_d, qdot), control_torques, diagram = run_simulation(
        q_initial=q_initial,
        q_final=q_final,
        B_avg=B_avg,
        K_p=K_p,
        K_d=K_d,
        simulation_duration_s=2.5,
        should_apply_control_torques=True,  # True or False?
    )
    # Convert `q`, `q_d`, `qdot`, and `qdot_d`  to degrees
    q_d = np.rad2deg(q_d)
    q = np.rad2deg(q)
    qdot_d = np.rad2deg(qdot_d)
    qdot = np.rad2deg(qdot)
    fig, ax = plot_diagram(diagram)
    fig.savefig('Block_Diagram_With_Control_Torques.png', dpi=300)
    

    # ----------------------------------------------------------------------------------
    # Animate Trajectory
    # TODO:
    #   Using the link lengths `[a_1, a_2]`, the actual joint positions `q` and desired
    #   joint positions `q_d`, with the `calc_fk_2D` function to calculate the xy
    #   positions of each joint of the manipulator for the actual and desired
    #   trajectories, respectively. (Replace `...` with the correct values)
    #
    #   Hint: Make sure to convert `q` back to radians before using it with `calc_fk_2D`
    #         (using np.deg2rad).
    #
    # ----------------------------------------------------------------------------------
    joint_xs, joint_ys = calc_fk_2D(link_lens=[a_1, a_2], joint_positions=np.deg2rad(q))
    joint_xs_desired, joint_ys_desired = calc_fk_2D(link_lens=[a_1, a_2], joint_positions=np.deg2rad(q_d))

    # ----------------------------------------------------------------------------------
    # TODO:
    #   Replace all `...` in the call of the `animate_2R_planar_arm_traj` function with
    #   the correct output of the `calc_fk_2D`.
    # ----------------------------------------------------------------------------------
    _, _, anim_control_torques = animate_2R_planar_arm_traj(
        joint_xs=joint_xs, 
        joint_ys=joint_ys,
        animation_file_name="control_torques_animation"
    )
    #writer = animation.FFMpegWriter(fps=3)
    #anim_control_torques.save('my_animation.mp4', writer=writer)

    # ----------------------------------------------------------------------------------
    # Plot Snapshots
    # TODO:
    #   Replace all `...` in the call of the `plot_snapshots` function with
    #   the correct output of the `calc_fk_2D` and the dt specified in the problem
    #   statement.
    #   Add code to properly label `ax` and save `fig`
    # ----------------------------------------------------------------------------------
    fig, ax = plot_snapshots(
        dt=0.1,
        joint_xs=joint_xs,
        joint_ys=joint_ys,
        joint_xs_desired=joint_xs_desired,
        joint_ys_desired=joint_ys_desired,
    )
    ax.set_title("Joint Animation With Control Torques")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.savefig("with_control_torques.png", dpi=300)


    # ----------------------------------------------------------------------------------
    # Plot Joint Position Error
    # TODO:
    #   Replace `...` with the code to make the specified joint position error plot for
    #   the inverse dynamics controller case.
    #
    # Hints:
    # 1. To plot a black dashed vertical line at `x = x0` use the `ax.axvline` function:
    #       `ax.axvline(x0, ls="--", color="black")
    # 2. When plotting, use the `label` argument to automatically add a legend item:
    #       `ax.plot(x, y, color="red", label=r"$\q_1$ Error")`
    # 3. You need to call `ax.legend()` to actually plot the legend.
    # ----------------------------------------------------------------------------------
    error_fig = plt.figure()
    error_plot = error_fig.add_subplot(1, 1, 1)
    error_plot.set_xlim(-0.1, 2.6)
    error_plot.set_ylim(-2.6, 2.6)
    error_plot.axhline(2, ls="--", color="k")
    error_plot.axhline(-2, ls="--", color="k")
    error_plot.plot(t, q[0,:]-q_d[0,:], color="r", label=r"$\tilde{q}_1$") 
    error_plot.plot(t, q[1,:]-q_d[1,:], color="b", label=r"$\tilde{q}_2$")
    error_plot.set_xlabel("Time [s]")
    error_plot.set_ylabel("Position Error [deg]")
    error_plot.set_title("Joint Position Errors")
    error_plot.legend()

    error_fig.savefig("joint_position_error.png", dpi=300)

    # ----------------------------------------------------------------------------------
    # Plot Joint Positions
    # TODO:
    #   Replace `...` with the code to make the specified joint position plot for the
    #   inverse dynamics controller case.
    # ----------------------------------------------------------------------------------
    position_fig = plt.figure()
    position_plot = position_fig.add_subplot(1, 1, 1)
    position_plot.set_xlim(-0.1, 2.6)
    position_plot.set_ylim(-360, 360)
    position_plot.axvline(1.25, ls="--", color="k")
    position_plot.plot(t, q_d[0,:], ls="--", color="r", label=r"$q_{1,d}$")
    position_plot.plot(t, q[0,:], color="r", label=r"$q_1$")
    position_plot.plot(t, q_d[1,:], ls="--", color="b", label=r"$q_{2,d}$")
    position_plot.plot(t, q[1,:], color="b", label=r"$q_2$")
    position_plot.set_xlabel("Time [s]")
    position_plot.set_ylabel("Position [deg]")
    position_plot.set_title("Joint Positions")
    position_plot.legend()
    
    position_fig.savefig("joint_positions.png", dpi=300)


   # ----------------------------------------------------------------------------------
    # Plot Joint Velocities
    # TODO:
    #   Replace `...` with the code to make the specified joint velocity plot for the
    #   inverse dynamics controller case.
    # ----------------------------------------------------------------------------------
    velocity_fig = plt.figure()
    velocity_plot = velocity_fig.add_subplot(1, 1, 1)
    velocity_plot.set_xlim(-0.1, 2.6)
    velocity_plot.set_ylim(-360, 360)
    velocity_plot.axvline(1.25, ls="--", color="k")
    velocity_plot.plot(t, qdot_d[0,:], ls="--", color="r", label=r"$\dot{q}_{1,d}$")
    velocity_plot.plot(t, qdot[0,:], color="r", label=r"$\dot{q}_1$")
    velocity_plot.plot(t, qdot_d[1,:], ls="--", color="b", label=r"$\dot{q}_{2,d}$")
    velocity_plot.plot(t, qdot[1,:], color="b", label=r"$\dot{q}_2$")
    velocity_plot.set_xlabel("Time [s]")
    velocity_plot.set_ylabel("Velocity [deg/s]")
    velocity_plot.set_title("Joint Velocities")
    velocity_plot.legend()
    
    velocity_fig.savefig("joint_velocities.png", dpi=300)

    # ----------------------------------------------------------------------------------
    # Plot Control Torques
    # TODO:
    #   Replace `...` with the code to make the specified control torque plot for the
    #   inverse dynamics controller case.
    # ----------------------------------------------------------------------------------
    torque_fig = plt.figure()
    torque_plot = torque_fig.add_subplot(1, 1, 1)
    #torque_plot.set_xlim(-0.1, 2.6)
    #torque_plot.set_ylim(-2.6, 2.6)
    torque_plot.axvline(1.25, ls="--", color="k")
    torque_plot.plot(t, control_torques[0,:], color="r", label=r"$\tau_1$") 
    torque_plot.plot(t, control_torques[1,:], color="b", label=r"$\tau_2$")
    torque_plot.set_xlabel("Time [s]")
    torque_plot.set_ylabel(" Torque [N*m]")
    torque_plot.set_title("Joint Torques")
    torque_plot.legend()

    torque_fig.savefig("joint_torques.png", dpi=300)
