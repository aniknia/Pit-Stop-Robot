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

# ------------------------------------------------------------------------------
# Global Variables
# ------------------------------------------------------------------------------
# Linkage Lengths
l1, l2, l3 = 0.15, 0.1, .07 #[m]
# ------------------------------------------------------------------------------

class InverseDynamicsController:
    def __init__(
        self,
        motor_group: DynamixelMotorGroup,
        K_P_out: NDArray[np.double],
        K_P_in: NDArray[np.double],
        K_D_out: NDArray[np.double],
        K_D_in: NDArray[np.double],
        K_I_out: NDArray[np.double],
        K_I_in: NDArray[np.double],
        q_initial_deg: Sequence[float],
        q_desired_deg: Sequence[float],
        qdot_initial_deg_per_s: Sequence[float],
        qdot_desired_deg_per_s: Sequence[float],
        qddot_desired_deg_per_s2: Sequence[float],
        max_duration_s: float = 3.0,
    ):
        # ------------------------------------------------------------------------------
        # Setting up Controller Related Variables
        # ------------------------------------------------------------------------------
        # Intial and Desired Positions
        self.q_initial_rad = np.deg2rad(q_initial_deg)
        self.q_desired_rad = np.deg2rad(q_desired_deg)

        # Intial and Desired Velocities
        self.qdot_initial_rad_per_s = np.deg2rad(qdot_initial_deg_per_s)
        self.qdot_desired_rad_per_s = np.deg2rad(qdot_desired_deg_per_s)

        # Intial and Desired Accelerations
        self.qddot_desired_rad_per_s2 = np.deg2rad(qddot_desired_deg_per_s2)

        # Gains
        self.K_P_out = np.asarray(K_P_out, dtype=np.double)
        self.K_P_in = np.asarray(K_P_in, dtype=np.double)
        self.K_D_out = np.asarray(K_D_out, dtype=np.double)
        self.K_D_in = np.asarray(K_D_in, dtype=np.double)
        self.K_I_out = np.asarray(K_I_out, dtype=np.double)
        self.K_I_in = np.asarray(K_I_in, dtype=np.double)

        self.control_freq_Hz = 30.0
        self.max_duration_s = float(max_duration_s)
        self.control_period_s = 1 / self.control_freq_Hz
        self.loop_manager = FixedFrequencyLoopManager(self.control_freq_Hz)
        self.should_continue = True

        self.joint_position_history = deque()
        self.time_stamps = deque()
        self.run_stamps = deque()
        self.endeff_position_history = deque()
        # ------------------------------------------------------------------------------


        # ------------------------------------------------------------------------------
        # Manipulator Parameters
        # ------------------------------------------------------------------------------
        # Density PLA, 30%
        rho = 1250*0.3 #[kg/m^3]
        # Linkage Lengths
        self.l1, self.l2, self.l3 = l1, l2, l3 #[m]
        # Center of Mass Lengths
        self.lc1, self.lc2, self.lc3 = 0.13, 0.08, 0.05 #[m]
        # Width of Link
        self.w1, self.w2, self.w3 = 0.5*(0.036+0.022), 0.5*(0.036+0.022), 0.5*(.036+0.023) #[m]
        # Masses
        self.m3 = 0.077 + 0.014*0.3 #[kg]
        self.m2 = self.m3 + 0.077 + self.w2*self.l2*0.003*rho*2 #[kg]
        self.m1 = self.m2 + 0.077 + self.w1*self.l1*0.004*rho*2 #[kg]
        # ------------------------------------------------------------------------------


        # ------------------------------------------------------------------------------
        # Motor Communication Related Variables
        # ------------------------------------------------------------------------------
        self.motor_group: DynamixelMotorGroup = motor_group
        # ------------------------------------------------------------------------------
    

        # ------------------------------------------------------------------------------
        # DC Motor Modeling
        # ------------------------------------------------------------------------------
        self.pwm_limits = []
        for info in self.motor_group.motor_info.values():
            self.pwm_limits.append(info.pwm_limit)
        self.pwm_limits = np.asarray(self.pwm_limits)

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

        positions = [self.q_desired_rad, self.q_initial_rad, self.q_desired_rad, self.q_initial_rad]
        velocities = [self.qdot_desired_rad_per_s, self.qdot_initial_rad_per_s, self.qdot_desired_rad_per_s, self.qdot_initial_rad_per_s]
        gains_kp = [self.K_P_out, self.K_P_in, self.K_P_out, self.K_P_in]
        gains_kd = [self.K_D_out, self.K_D_in, self.K_D_out, self.K_D_in]
        gains_ki = [self.K_I_out, self.K_I_in, self.K_I_out, self.K_I_in]

        start_time = time.time()

        for i, x in enumerate(positions):

            run_time = time.time()
            integral_error=0

            while self.should_continue:
                # --------------------------------------------------------------------------
                # Step 1 - Get feedback
                # --------------------------------------------------------------------------
                # Position Feedback (Actual)
                q_rad = np.asarray(list(self.motor_group.angle_rad.values()))

                # Velocity Feedback (Actual)
                qdot_rad_per_s = np.asarray(list(self.motor_group.velocity_rad_per_s.values()))

                # Save for plotting
                self.joint_position_history.append(q_rad)
                self.endeff_position_history.append(self.joints2endeff(q_rad))
                self.time_stamps.append(time.time() - start_time)
                self.run_stamps.append(time.time() - run_time)
                # --------------------------------------------------------------------------


                # --------------------------------------------------------------------------
                # Step 2 - Check termination criterion
                # --------------------------------------------------------------------------
                # Stop after max_duration_s seconds
                if self.run_stamps[-1] - self.run_stamps[0] > self.max_duration_s:
                    break
                # --------------------------------------------------------------------------

                # --------------------------------------------------------------------------
                # Step 3 - Outer Control Loop
                # --------------------------------------------------------------------------
                # Position Error
                q_error = x - q_rad

                # Velocity Error
                qdot_error = velocities[i] - qdot_rad_per_s

                # Integral Error for Steady State
                integral_error += q_error * self.control_period_s

                y = (gains_kp[i] @ q_error) + (gains_kd[i] @ qdot_error) + (gains_ki[i] @ integral_error) + self.qddot_desired_rad_per_s2
                # --------------------------------------------------------------------------

                # --------------------------------------------------------------------------
                # Step 4 - Inner Control Loop
                # --------------------------------------------------------------------------
                #Inertia Matrix
                B_q = self.compute_inertia_matrix(q_rad)

                #Nonlinear Components
                n = (self.compute_coriolis_matrix(q_rad, qdot_rad_per_s) @ qdot_rad_per_s) + self.calc_gravity_compensation_torque(q_rad)

                #Torque Output Controls
                u = (B_q @ y) + n
                # --------------------------------------------------------------------------


                # --------------------------------------------------------------------------
                # Step 5 - Command control action
                # --------------------------------------------------------------------------
                # Converts the torque control action into a PWM command
                pwm_command = self.motor_model.calc_pwm_command(u)

                self.motor_group.pwm = {
                    dxl_id: pwm_value
                    for dxl_id, pwm_value in zip(
                        self.motor_group.dynamixel_ids, pwm_command, strict=True
                    )
                }
                # --------------------------------------------------------------------------

                # Creates fixed frequency for while loop
                self.loop_manager.sleep()

        self.stop()

    def stop(self):
        self.should_continue = False
        time.sleep(2 * self.control_period_s)
        self.motor_group.disable_torque()

    def signal_handler(self, *_):
        self.stop()

    def compute_inertia_matrix(
        self, joint_positions_rad: NDArray[np.double]
    ) -> NDArray[np.double]:
        q1, q2, q3 = joint_positions_rad

        m1, m2, m3 = self.m1, self.m2, self.m3
        l1, l2, l3 = self.l1, self.l2, self.l3
        lc1, lc2, lc3 = self.lc1, self.lc2, self.lc3
        w1, w2, w3 = self.w1, self.w2, self.w3

        #Moment of Inertia of each link, added these values to the B matrix for 2d rectangular inertia about perpendicular axis
        I1 = 1/12 * m1 * (l1**2 + w1**2)
        I2 = 1/12 * m2 * (l2**2 + w2**2)
        I3 = 1/12 * m3 * (l3**2 + w3**2)

        #Initialize Inertia Matrix
        B_q = np.zeros((3, 3))

        # Diagonal Terms
        B_q[0, 0] = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(q2)) + m3 * (l1**2 + l2**2 + lc3**2 + 2 * l1 * l2 * np.cos(q2) + 2 * l1 * lc3 * np.cos(q2 + q3) + 2 * l2 * lc3 * np.cos(q3))
        B_q[1, 1] = m2 * lc2**2 + m3 * (l2**2 + lc3**2 + 2 * l2 * lc3 * np.cos(q3))
        B_q[2, 2] = m3 * lc3**2

        # Off-diagonal Terms (symmetric)
        B_q[0, 1] = B_q[1, 0] = m2 * (lc2**2 + l1 * lc2 * np.cos(q2)) + m3 * (l2**2 + lc3**2 + l1 * l2 * np.cos(q2) + l1 * lc3 * np.cos(q2 + q3) + 2 * l2 * lc3 * np.cos(q3))
        B_q[0, 2] = B_q[2, 0] = m3 * (lc3**2 + l2 * lc3 * np.cos(q3) + l1 * lc3 * np.cos(q2 + q3))
        B_q[1, 2] = B_q[2, 1] = m3 * (lc3**2 + l2 * lc3 * np.cos(q3))

        return B_q

    def compute_coriolis_matrix(
        self, joint_positions_rad, joint_velocities_rad: NDArray[np.double]
    ) -> NDArray[np.double]:
        q1, q2, q3 = joint_positions_rad
        qdot1, qdot2, qdot3 = joint_velocities_rad

        m1, m2, m3 = self.m1, self.m2, self.m3
        l1, l2 = self.l1, self.l2
        lc1, lc2, lc3 = self.lc1, self.lc2, self.lc3

        #Initialize Inertia Matrix
        C = np.zeros((3, 3))

        #Christoffel Symbols
        c12 = m2*l1*lc2
        c23 = m3*l1*lc3
        c13 = m3*l1*lc3
        c123 = m2*l1*lc2 + m3*l1*l2

        #Coriolis matrix, C33 = 0
        C[0, 0] = -c123*np.sin(q2)*qdot2 - c23*np.sin(q3)*qdot3 - c13*np.sin(q2+q3)*(qdot2+qdot3)
        C[0, 1] = -c123*np.sin(q2)*(qdot1+qdot2) - c23*np.sin(q3)*qdot3 - c13*np.sin(q2+q3)*(qdot1+qdot2+qdot3)
        C[0, 2] = -(c23*np.sin(q3) - c13*np.sin(q2+q3))*(qdot1+qdot2+qdot3)
        C[1, 0] = c123*np.sin(q2)*qdot1 + c13*np.sin(q2+q3)*(qdot1+qdot2+qdot3)
        C[1, 1] = -(c23*np.sin(q3) + c13*np.sin(q2+q3))*qdot3
        C[1, 2] = -(c23*np.sin(q3) + c13*np.sin(q2+q3))*(qdot1+qdot2+qdot3)
        C[2, 0] = c23*np.sin(q3)*qdot1 + c13*np.sin(q2+q3)*(qdot1+qdot2+qdot3)
        C[2, 1] = c23*np.sin(q3)*qdot2 + c13*np.sin(q2+q3)*(qdot2+qdot3)

        return C

    def calc_gravity_compensation_torque(
        self, joint_positions_rad: NDArray[np.double]
    ) -> NDArray[np.double]:
        q1, q2, q3 = joint_positions_rad
      
        from math import cos
        g = 9.81

        m1, m2, m3 = self.m1, self.m2, self.m3
        l1, l2 = self.l1, self.l2
        lc1, lc2, lc3 = self.lc1, self.lc2, self.lc3

        print(g*(m2*lc2*cos(q1+q2) + m3*(l2*cos(q1+q2) + lc3*cos(q1+q2+q3))))

        return np.array(
            [
                g*(m1*lc1*cos(q1) + m2*(l1*cos(q1) + lc2*cos(q1+q2)) + m3*(l1*cos(q1) + l2*cos(q1+q2) + lc3*cos(q1+q2+q3))),
                g*(m2*lc2*cos(q1+q2) + m3*(l2*cos(q1+q2) + lc3*cos(q1+q2+q3)) - 0.01),
                g*m3*lc3*cos(q1+q2+q3)
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
        abs_tol = math.radians(2.0)

        should_continue_loop = True
        while should_continue_loop:
            should_continue_loop = False
            q_rad = self.motor_group.angle_rad
            for dxl_id in home_positions_rad:
                if abs(home_positions_rad[dxl_id] - q_rad[dxl_id]) > abs_tol:
                    should_continue_loop = True
                    break
        
        # Waits 2 seconds before continuing
        time.sleep(2)
        
        # Set PWM Mode (i.e. voltage control)
        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.PWM)
        self.motor_group.enable_torque()

    def joints2endeff(
        self, joint_positions_rad: NDArray[np.double]
    ) -> NDArray[np.double]:
        q1, q2, q3 = joint_positions_rad
        #Forward kinematics from joint

        # Recover theta1, theta2, theta3 (flipped motors)
        theta1 = 2 * np.pi - q1
        theta2 = 2 * np.pi - q2
        theta3 = 2 * np.pi - q3
        
        theta1eff = theta1 - np.pi
        theta2eff = np.pi - (theta1eff + theta2)

        # Compute (x0, y0) using forward kinematics for the first two links
        x0 = l1 * np.cos(theta1eff) + l2 * np.cos(theta2eff)
        y0 = l1 * np.sin(theta1eff) - l2 * np.sin(theta2eff)
        
        # Compute tilt
        tilt = (theta1eff + theta2 + theta3) - 2*np.pi
        
        # Compute end-effector position (x, y)
        x = x0 + l3 * np.cos(tilt)
        y = y0 + l3 * np.sin(tilt)

        return [x, y]

if __name__ == "__main__":
    
    # Inverse Kinematics
    # Input end effector x, y, and tilt angle
    def endeff2joints(x, y, tilt):

        tilt = np.deg2rad(tilt)
        
        x0 = x - l3*np.cos(tilt)
        y0 = y - l3*np.sin(tilt)
        
        theta1A = np.arctan(y0/x0)
        theta1B = np.arccos((l1**2 + x0**2 + y0**2 - l2**2)/(2*l1*np.sqrt(x0**2 + y0**2)))
        
        theta1 = np.pi + theta1A + theta1B
        theta2 = np.arccos((l2**2 + l1**2 - x0**2 - y0**2)/(2*l2*l1))
        theta3 = 2*np.pi - (theta1A + theta1B) - theta2 + tilt

        # Flips motors (orientation)
        q1 = 2*np.pi - theta1
        q2 = 2*np.pi - theta2
        q3 = 2*np.pi - theta3

        return np.rad2deg([q1, q2, q3])
    
    # Initial Position
    q_initial_endeff = [0.18,0.05,0]
    q_initial = endeff2joints(*q_initial_endeff)
    # Desired Position
    q_desired_endeff = [0.3,0.05,0]
    q_desired = endeff2joints(*q_desired_endeff)

    # Initial Joint Velocities
    qdot_initial = [0, 0, 0]
    # Desired Joint Velocities
    qdot_desired = [0, 0, 0]

    # Desired Joint Acceleration
    qddot_desired = [0, 0, 0]

    # Desired Gains
    # Proportional Gain desired
    K_P_out = np.array([[211, 0, 0],
                   [0, 420, 0],
                   [0, 0, 2100]])
    # Derivative Gain desired
    K_D_out = np.array([[13.8, 0, 0],
                   [0, 60, 0],
                   [0, 0, 2]])
    # Integral Gain desired
    K_I_out = np.array([[200, 0, 0],
                   [0, 280, 0],
                   [0, 0, 60]])
    

    # Initial Gains
    # Proportional Gain Initial
    K_P_in = np.array([[65, 0, 0],
                   [0, 70, 0],
                   [0, 0, 1200]])

    # Derivative Gain Initial
    K_D_in = np.array([[9.2, 0, 0],
                   [0, 32, 0],
                   [0, 0, 2]])
    
    # Integral Gain Initial
    K_I_in = np.array([[160, 0, 0],
                   [0, 750, 0],
                   [0, 0, 600]])
    
    """ BEFORE ADDING NONLINEAR TERMS
    # Proportional Gain
    K_P = np.array([[3, 0, 0],
                   [0, 1.85, 0],
                   [0, 0, 1.75]])

    # Derivative Gain
    K_D = np.array([[0.16, 0, 0],
                   [0, 0.04, 0],
                   [0, 0, 0.13]])
    """

    # Correct COM Port and Baud Rate
    # Mac: /dev/tty.usbserial-FT9BTFVF
    # Windows: COM4
    dxl_io = DynamixelIO(
        device_name="COM4",
        baud_rate=57_600,
    )

    # Create `DynamixelMotorFactory` object to create dynamixel motor object
    motor_factory = DynamixelMotorFactory(
        dxl_io=dxl_io,
        dynamixel_model=DynamixelModel.MX28
    )

    # Correct Motor IDs
    dynamixel_ids = 0, 1, 2

    motor_group = motor_factory.create(*dynamixel_ids)

    # Make controller
    controller = InverseDynamicsController(
        motor_group=motor_group,
        K_P_out=K_P_out,
        K_P_in=K_P_in,
        K_D_out=K_D_out,
        K_D_in=K_D_in,
        K_I_out=K_I_out,
        K_I_in=K_I_in,
        q_initial_deg=q_initial,
        q_desired_deg=q_desired,
        qdot_initial_deg_per_s=qdot_initial,
        qdot_desired_deg_per_s=qdot_desired,
        qddot_desired_deg_per_s2=qddot_desired
    )
    # ----------------------------------------------------------------------------------

    # Run controller
    controller.start_control_loop()

    # Extract results
    time_stamps = np.asarray(controller.time_stamps)
    joint_positions = np.rad2deg(controller.joint_position_history).T
    endeff_positions = np.asarray(controller.endeff_position_history).T

    # ----------------------------------------------------------------------------------
    # Plot Results
    # ----------------------------------------------------------------------------------

    # Joint Angles vs Time
    # Create figure and axes
    fig, (ax_motor0, ax_motor1, ax_motor2) = plt.subplots(3, 1, figsize=(10, 12))

    # Label Plots
    fig.suptitle(f"Motor Angles vs Time")
    ax_motor0.set_title(f"Motor Joint 0 (KP: {K_P_out[0,0]}, {K_P_in[0,0]} KD: {K_D_out[0,0]}, {K_D_in[0,0]} KI: {K_I_out[0,0]}, {K_I_in[0,0]})")
    ax_motor1.set_title(f"Motor Joint 1 (KP: {K_P_out[1,1]}, {K_P_in[1,1]} KD: {K_D_out[1,1]}, {K_D_in[1,1]} KI: {K_I_out[1,1]}, {K_I_in[1,1]})")
    ax_motor2.set_title(f"Motor Joint 2 (KP: {K_P_out[2,2]}, {K_P_in[2,2]} KD: {K_D_out[2,2]}, {K_D_in[2,2]} KI: {K_I_out[2,2]}, {K_I_in[2,2]})")

    ax_motor2.set_xlabel("Time [s]")

    ax_motor0.set_ylabel("Angle [deg]")
    ax_motor1.set_ylabel("Angle [deg]")
    ax_motor2.set_ylabel("Angle [deg]")

    ax_motor0.axhline(
        math.degrees(controller.q_desired_rad[0]), 
        ls="--", 
        color="red", 
        label="Setpoint"
    )
    ax_motor0.axhline(
        math.degrees(controller.q_initial_rad[0]), 
        ls="--", 
        color="red"
    )
    ax_motor1.axhline(
        math.degrees(controller.q_desired_rad[1]), 
        ls="--", 
        color="red", 
        label="Setpoint"
    )
    ax_motor1.axhline(
        math.degrees(controller.q_initial_rad[1]), 
        ls="--", 
        color="red"
    )
    ax_motor2.axhline(
        math.degrees(controller.q_desired_rad[2]), 
        ls="--", 
        color="red", 
        label="Setpoint"
    )
    ax_motor2.axhline(
        math.degrees(controller.q_initial_rad[2]), 
        ls="--", 
        color="red"
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
    ax_motor2.axhline(
        math.degrees(controller.q_desired_rad[2]) - 1, 
        ls=":", 
        color="blue", 
        label="Convergence Bound"
    )
    ax_motor2.axhline(
        math.degrees(controller.q_desired_rad[2]) + 1, ls=":", color="blue"
    )
    ax_motor2.axvline(1.5, ls=":", color="purple")

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
    ax_motor2.plot(
        time_stamps,
        joint_positions[2],
        color="black",
        label="Motor Angle Trajectory",
    )

    ax_motor0.legend()
    ax_motor1.legend()
    ax_motor2.legend()

    fig.savefig("joint angles.png")

    # End Effector vs Time
    fig, (x_axis, y_axis) = plt.subplots(2, 1, figsize=(10, 12))

    # Label Plot
    fig.suptitle(f"End Effector Position vs Time")
    x_axis.set_title("X Axis")
    y_axis.set_title("Y Axis")

    y_axis.set_xlabel("Time [s]")

    x_axis.set_ylabel("X Position [m]")
    y_axis.set_ylabel("Y Position [m]")

    # Plot end effector positions
    x_axis.plot(
        time_stamps,
        endeff_positions[0],
        color="black",
        label="End Effector Position",
    )
    y_axis.plot(
        time_stamps,
        endeff_positions[1],
        color="black",
        label="End Effector Position",
    )
    x_axis.axhline(
        q_desired_endeff[0], 
        ls="--", 
        color="red", 
        label="Setpoint"
    )
    x_axis.axhline(
        q_desired_endeff[0] - 0.0025, 
        ls=":", 
        color="blue", 
        label="Convergence Bound"
    )
    x_axis.axhline(
        q_desired_endeff[0] + 0.0025, 
        ls=":", 
        color="blue", 
        label="Convergence Bound"
    )
    y_axis.axhline(
        q_desired_endeff[1], 
        ls="--", 
        color="red", 
        label="Setpoint"
    )
    y_axis.axhline(
        q_desired_endeff[1] - 0.0025, 
        ls=":", 
        color="blue", 
        label="Convergence Bound"
    )
    y_axis.axhline(
        q_desired_endeff[1] + 0.0025, 
        ls=":", 
        color="blue", 
        label="Convergence Bound"
    )

    x_axis.legend()
    y_axis.legend()

    fig.savefig("end effector positions.png")

    # ----------------------------------------------------------------------------------
    plt.show()