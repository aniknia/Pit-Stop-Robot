import math
import signal
import time
from collections import deque
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from dxl import (
    DynamixelMode, 
    DynamixelModel, 
    DynamixelMotorGroup, 
    DynamixelMotorFactory, 
    DynamixelIO
)
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from mechae263C_helpers.minilabs import FixedFrequencyLoopManager, DCMotorModel


class PDwGravityCompensationController:
    def __init__(
        self,
        motor_group: DynamixelMotorGroup,
        K_P: NDArray[np.double],
        K_D: NDArray[np.double],
        q_initial_deg: Sequence[float],
        q_desired_deg: Sequence[float],
        max_duration_s: float = 2.0,
    ):
        # ------------------------------------------------------------------------------
        # Controller Related Variables
        # ------------------------------------------------------------------------------
        self.q_initial_rad = np.deg2rad(q_initial_deg)
        self.q_desired_rad = np.deg2rad(q_desired_deg)

        self.K_P = np.asarray(K_P, dtype=np.double)
        self.K_D = np.asarray(K_D, dtype=np.double)

        self.control_freq_Hz = 30.0
        self.max_duration_s = float(max_duration_s)
        self.control_period_s = 1 / self.control_freq_Hz
        self.loop_manager = FixedFrequencyLoopManager(self.control_freq_Hz)
        self.should_continue = True

        self.joint_position_history = deque()
        self.time_stamps = deque()
        # ------------------------------------------------------------------------------


        # ------------------------------------------------------------------------------
        # Manipulator Parameters
        # ------------------------------------------------------------------------------
        self.m1, self.m2 = 0.193537, 0.0156075
        self.lc1, self.lc2 = 0.0533903, 0.0281188
        self.l1 = 0.0675
        # ------------------------------------------------------------------------------


        # ------------------------------------------------------------------------------
        # Motor Communication Related Variables
        # ------------------------------------------------------------------------------
        self.motor_group: DynamixelMotorGroup = motor_group
    

        # ------------------------------------------------------------------------------
        # DC Motor Modeling
        # ------------------------------------------------------------------------------
        self.pwm_limits = []
        for info in self.motor_group.motor_info.values():
            self.pwm_limits.append(info.pwm_limit)
        self.pwm_limits = np.asarray(self.pwm_limits)

        # This model is based on the DC motor model learned in class, it allows us to
        # convert the torque control action u into something we can actually send to the
        # MX28-AR dynamixel motors (pwm voltage commands).
        self.motor_model = DCMotorModel(
            self.control_period_s, pwm_limits=self.pwm_limits
        )
        # ------------------------------------------------------------------------------


        # ------------------------------------------------------------------------------
        # Clean Up / Exit Handler Code
        # ------------------------------------------------------------------------------
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        # ------------------------------------------------------------------------------
        
    def start_control_loop(self):
        self.go_to_home_configuration()

        start_time = time.time()
        while self.should_continue:
            # --------------------------------------------------------------------------
            # Step 1 - Get feedback
            # --------------------------------------------------------------------------
            # Read position feedback (and covert resulting dict into to NumPy array)
            q_rad = np.asarray(list(self.motor_group.angle_rad.values()))

            # TODO: Read Data from Multiple Dynamixels – Joint Velocities (Question 2)
            #    Use the example above for retreiving joint position feedback (`q_rad`)
            #    and the `DynamixelMotorGroup.velocity_rad_per_s` property to extract 
            #    the joint velocity feedback in units of rad/s.
            qdot_rad_per_s = (
                np.asarray(list(...))
            )

            self.joint_position_history.append(q_rad)  # Save for plotting
            self.time_stamps.append(time.time() - start_time)  # Save for plotting
            # --------------------------------------------------------------------------


            # --------------------------------------------------------------------------
            # Step 2 - Check termination criterion
            # --------------------------------------------------------------------------
            # Stop after 2 seconds
            if self.time_stamps[-1] - self.time_stamps[0] > self.max_duration_s:
                self.stop()
                return
            # --------------------------------------------------------------------------


            # --------------------------------------------------------------------------
            # Step 3 - Compute error term
            # --------------------------------------------------------------------------
            # TODO: Compute Error Term (Question 3)
            # Use the `self.q_desired` variable and the `q_actual` variable to compute
            # the joint position error for the current time step.
            q_error = ...
            # --------------------------------------------------------------------------


            # --------------------------------------------------------------------------
            # Step 4 - Calculate control action
            # --------------------------------------------------------------------------
            gravity_comp_torques = self.calc_gravity_compensation_torque(q_rad)

            # TODO: Calculate Control Law (Question 3)
            # Use the `self.K_P`, `q_error`, `self.K_D`, `q_dot_actual`, and
            # `gravity_comp_torques` variables to compute the control action for joint
            # space PD control with gravity compensation.
            #
            # Note: This is a torque control action!
            u = ...
            # --------------------------------------------------------------------------


            # --------------------------------------------------------------------------
            # Step 5 - Command control action
            # --------------------------------------------------------------------------
            # This code converts the torque control action into a PWM command using a
            # model of the dynamixel motors
            pwm_command = self.motor_model.calc_pwm_command(u)

            # TODO:  Sending Joint PWM Commands (Question 4)
            # Replace "..." with the calculated `pwm_command` variable
            self.motor_group.pwm = {
                dxl_id: pwm_value
                for dxl_id, pwm_value in zip(
                    self.motor_group.dynamixel_ids, ..., strict=True
                )
            }
            # --------------------------------------------------------------------------


            # Print current position in degrees
            print("q [deg]:", np.degrees(q_rad))

            # This code helps this while loop run at a fixed frequency
            self.loop_manager.sleep()

        self.stop()

    def stop(self):
        self.should_continue = False
        time.sleep(2 * self.control_period_s)
        self.motor_group.disable_torque()

    def signal_handler(self, *_):
        self.stop()

    def calc_gravity_compensation_torque(
        self, joint_positions_rad: NDArray[np.double]
    ) -> NDArray[np.double]:
        q1, q2 = joint_positions_rad
      
        from math import cos
        g = 9.81

        m1, m2 = self.m1, self.m2
        l1 = self.l1
        lc1, lc2 = self.lc1, self.lc2

        return -np.array(
            [
                m1 * g * lc1 * cos(q1) + m2 * g * (l1 * cos(q1) + lc2 * cos(q1 + q2)),
                m2 * g * lc2 * cos(q1 + q2)
            ]
        ) 


    def go_to_home_configuration(self):
        """Puts the motors in 'home' position"""
        self.should_continue = True
        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.Position)
        self.motor_group.enable_torque()

        # Move to home position (self.q_initial)
        home_positions_rad = {
            dynamixel_id: self.q_initial_rad[i]
            for i, dynamixel_id in enumerate(self.motor_group.dynamixel_ids)
        }
        
        self.motor_group.angle_rad = home_positions_rad
        time.sleep(0.5)
        abs_tol = math.radians(1.0)
        
        should_continue_loop = True
        while should_continue_loop:
            should_continue_loop = False
            q_rad = self.motor_group.angle_rad
            for dxl_id in home_positions_rad:
                if abs(home_positions_rad[dxl_id] - q_rad[dxl_id]) > abs_tol:
                    should_continue_loop = True
                    break
            

        # Set PWM Mode (i.e. voltage control)
        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.PWM)
        self.motor_group.enable_torque()


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # TODO: Tuning Controller Gains and Plot (Question 5)
    # 1) Replace the corresponding `...` values with the initial and desired joint
    #    configurations.
    # 2) Replace the corresponding `...` values with your K_P and K_D gain matrices
    # 3) Replace the corresponding `...` with the IDs of the dynamixel motors used in
    #    your manipulator.
    # ----------------------------------------------------------------------------------
    # A Python list with two elements representing the initial joint configuration
    q_initial = [..., ...]

    # A Python list with two elements representing the desired joint configuration
    q_desired = [..., ...]

    # A numpy array of shape (2, 2) representing the proportional gains of your
    # controller
    K_P = ...

    # A numpy array of shape (2, 2) representing the derivative gains of your controller
    K_D = ...

    # # A tuple with two elements representing the IDs of the dynamixels used in your
    # manipulator
    # ----------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------
    # Create `DynamixelIO` object to store the serial connection to U2D2
    #
    # TODO: Replace "..." below with the correct serial port found from Dynamixel Wizard
    #
    # Note: You may need to change the Baud Rate to match the value found from
    #       Dynamixel Wizard
    dxl_io = DynamixelIO(
        device_name="...",
        baud_rate=57_600,
    )

    # Create `DynamixelMotorFactory` object to create dynamixel motor object
    motor_factory = DynamixelMotorFactory(
        dxl_io=dxl_io,
        dynamixel_model=DynamixelModel.MX28
    )

    # TODO: Replace "..." below with the correct Dynamixel IDs found from Dynamixel Wizard 
    #       (in order of closest to base frame first)
    dynamixel_ids = ..., ...

    motor_group = motor_factory.create(*dynamixel_ids)

    # Make controller
    controller = PDwGravityCompensationController(
        motor_group=motor_group,
        K_P=K_P,
        K_D=K_D,
        q_initial_deg=q_initial,
        q_desired_deg=q_desired
    )
    # ----------------------------------------------------------------------------------

    # Run controller
    controller.start_control_loop()

    # Extract results
    time_stamps = np.asarray(controller.time_stamps)
    joint_positions = np.rad2deg(controller.joint_position_history).T


    # ----------------------------------------------------------------------------------
    # Plot Results
    # TODO: Section 5 (Plotting Joint Position Time Histories)
    # Plot joint positions of the manipulator versus time using the `joint_positions`
    # and `time_stamps` variables, respectively.
    # ----------------------------------------------------------------------------------
    date_str = datetime.now().strftime("%d-%m_%H-%M-%S")
    fig_file_name = f"joint_positions_vs_time_{date_str}.pdf"

    # Create figure and axes
    fig = plt.figure(figsize=(10, 5))
    ax_motor0 = fig.add_subplot(121)
    ax_motor1 = fig.add_subplot(122)

    # Label Plots
    fig.suptitle(f"Motor Angles vs Time")
    ax_motor0.set_title("Motor Joint 0")
    ax_motor1.set_title("Motor Joint 1")
    ax_motor0.set_xlabel("Time [s]")
    ax_motor1.set_xlabel("Time [s]")
    ax_motor0.set_ylabel("Motor Angle [deg]")
    ax_motor1.set_ylabel("Motor Angle [deg]")

    ax_motor0.axhline(
        math.degrees(controller.q_desired_rad[0]), 
        ls="--", 
        color="red", 
        label="Setpoint"
    )
    ax_motor1.axhline(
        math.degrees(controller.q_desired_rad[1]), 
        ls="--", 
        color="red", 
        label="Setpoint"
    )
    ax_motor0.axhline(
        math.degrees(controller.q_desired_rad[0]) - 1, ls=":", color="blue"
    )
    ax_motor0.axhline(
        math.degrees(controller.q_desired_rad[0]) + 1, 
        ls=":", 
        color="blue", 
        label="Convergence Bound"
    )
    ax_motor0.axvline(1.5, ls=":", color="purple")
    ax_motor1.axhline(
        math.degrees(controller.q_desired_rad[1]) - 1, 
        ls=":", 
        color="blue", 
        label="Convergence Bound"
    )
    ax_motor1.axhline(
        math.degrees(controller.q_desired_rad[1]) + 1, ls=":", color="blue"
    )
    ax_motor1.axvline(1.5, ls=":", color="purple")

    # Plot motor angle trajectories
    ax_motor0.plot(
        time_stamps,
        joint_positions[0],
        color="black",
        label="Motor Angle Trajectory",
    )
    ax_motor1.plot(
        time_stamps,
        joint_positions[1],
        color="black",
        label="Motor Angle Trajectory",
    )
    ax_motor0.legend()
    ax_motor1.legend()
    fig.savefig(fig_file_name)
    # ----------------------------------------------------------------------------------
    plt.show()
