import math
import time
import sys  # for exiting if sim connection fails
from collections import deque
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import sim  # CoppeliaSim Legacy Remote API
import simConst
from matplotlib import pyplot as plt
from numpy.typing import NDArray
# Linkage Lengths
l1, l2, l3 = 0.15, 0.1, .07 #[m]

class SimulatedMotorGroup:
    def __init__(self, joint_names: list[str], client_id: int):
        self.client_id = client_id
        self.joint_names = joint_names
        self.motor_handles = []

        # Get and store joint handles
        for name in joint_names:
            err_code, handle = sim.simxGetObjectHandle(client_id, name, sim.simx_opmode_blocking)
            if err_code != sim.simx_return_ok:
                raise RuntimeError(f"Failed to get handle for joint: {name}")
            self.motor_handles.append(handle)
            sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_streaming)

        self._last_angles = [0.0] * len(joint_names)
        self._last_time = time.time()

    @property
    def angle_rad(self):
        angles = []
        for handle in self.motor_handles:
            err_code, angle = sim.simxGetJointPosition(self.client_id, handle, sim.simx_opmode_buffer)
            angles.append(angle if err_code == sim.simx_return_ok else 0.0)
        self._last_angles = angles
        return {i: angle for i, angle in enumerate(angles)}

    @property
    def velocity_rad_per_s(self):
        current_time = time.time()
        new_angles = self.angle_rad
        dt = current_time - self._last_time
        self._last_time = current_time
        velocities = {
            i: (new_angles[i] - self._last_angles[i]) / dt
            for i in new_angles
        }
        return velocities

    @property
    def motor_info(self):
        return {i: type("MotorInfo", (), {"pwm_limit": 885})() for i in range(len(self.motor_handles))}

    @property
    def pwm(self):
        raise NotImplementedError("Use pwm.setter to send position commands.")

    @pwm.setter
    def pwm(self, pwm_dict):
        for i, pwm_value in pwm_dict.items():
            sim.simxSetJointTargetPosition(self.client_id, self.motor_handles[i], pwm_value, sim.simx_opmode_oneshot)

    def disable_torque(self):
        pass  # Stubbed for compatibility

    def enable_torque(self):
        pass  # Stubbed for compatibility

    def set_mode(self, mode):
        pass  # Stubbed for compatibility

class FixedFrequencyLoopManager:
    def __init__(self, freq_Hz):
        self.period = 1.0 / freq_Hz
        self.next_time = time.time() + self.period

    def sleep(self):
        now = time.time()
        if self.next_time > now:
            time.sleep(self.next_time - now)
        self.next_time += self.period


class InverseDynamicsController:
    def __init__(
        self,
        motor_group,  # ✅ Accept either SimulatedMotorGroup or DynamixelMotorGroup-like interface
        K_P: NDArray[np.double],
        K_D: NDArray[np.double],
        K_I: NDArray[np.double],
        q_initial_deg: Sequence[float],
        q_desired_deg: Sequence[float],
        qdot_initial_deg_per_s: Sequence[float],
        qdot_desired_deg_per_s: Sequence[float],
        qddot_desired_deg_per_s2: Sequence[float],
        max_duration_s: float = 8.0,
    ):
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
        self.K_P = np.asarray(K_P, dtype=np.double)
        self.K_D = np.asarray(K_D, dtype=np.double)
        self.K_I = np.asarray(K_I, dtype=np.double)

        self.control_freq_Hz = 30.0
        self.max_duration_s = float(max_duration_s)
        self.control_period_s = 1 / self.control_freq_Hz
        self.loop_manager = FixedFrequencyLoopManager(self.control_freq_Hz)
        self.should_continue = True

        self.joint_position_history = deque()
        self.time_stamps = deque()
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
        # --------------------------------------------------------------------------
        # Motor Communication (Simulated Motor Group)
        # --------------------------------------------------------------------------
        self.motor_group = motor_group  # ✅ Keep this; now it's your SimulatedMotorGroup
                # --------------------------------------------------------------------------

    def start_control_loop(self):
        self.go_to_home_configuration()  # ✅ Assumes this method works with simulated motors

        start_time = time.time()
        integral_error = 0

        while self.should_continue:
            # --------------------------------------------------------------------------
            # Step 1 - Get feedback
            # --------------------------------------------------------------------------
            # ✅ Simulated Joint Positions
            q_rad = np.asarray(list(self.motor_group.angle_rad.values()))

            # ✅ Simulated Velocities (estimated in SimulatedMotorGroup)
            qdot_rad_per_s = np.asarray(list(self.motor_group.velocity_rad_per_s.values()))

            # ✅ Save for plotting
            self.joint_position_history.append(q_rad)
            self.time_stamps.append(time.time() - start_time)
            # --------------------------------------------------------------------------
                # Step 2 - Check termination criterion
            if self.time_stamps[-1] - self.time_stamps[0] > self.max_duration_s:
                self.stop()
                return

                # Step 3 - Outer Control Loop
            q_error = self.q_desired_rad - q_rad
            qdot_error = self.qdot_desired_rad_per_s - qdot_rad_per_s
            integral_error += q_error * self.control_period_s

            y = (self.K_P @ q_error) + (self.K_D @ qdot_error) + (self.K_I @ integral_error) + self.qddot_desired_rad_per_s2

                # Step 4 - Inner Control Loop
            B_q = self.compute_inertia_matrix(q_rad)
            n = (self.compute_coriolis_matrix(q_rad, qdot_rad_per_s) @ qdot_rad_per_s) + self.calc_gravity_compensation_torque(q_rad)
            u = (B_q @ y) + n

                # Step 5 - Command control action
            pwm_command = self.motor_model.calc_pwm_command(u)

                # ✅ This works with SimulatedMotorGroup since we mimic `.pwm` setter
            self.motor_group.pwm = {
                dxl_id: pwm_value
                for dxl_id, pwm_value in zip(
                    self.motor_group.dynamixel_ids, pwm_command, strict=True
                )
            }

                # Enforce timing
            self.loop_manager.sleep()

                # On exit
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
            """Moves simulated joints to the home position."""
            self.should_continue = True
    
            # Prepare dictionary of joint indices and target angles
            home_positions_rad = {
                joint_idx: self.q_initial_rad[joint_idx]
                for joint_idx in self.motor_group.dynamixel_ids
            }
    
            # Send target positions via simulated motor group
            for joint_idx, pos in home_positions_rad.items():
                self.motor_group._last_angles[joint_idx] = pos  # track internally
                self.motor_group.pwm = {joint_idx: pos}         # sends to CoppeliaSim
    
            time.sleep(0.5)  # allow simulation time to move
    
            # Wait for joints to reach near target
            abs_tol = math.radians(2.0)
            should_continue_loop = True
    
            while should_continue_loop:
                should_continue_loop = False
                q_rad = self.motor_group.angle_rad
                for joint_idx in home_positions_rad:
                    if abs(home_positions_rad[joint_idx] - q_rad[joint_idx]) > abs_tol:
                        should_continue_loop = True
                        break
    
            time.sleep(2)  # allow for stabilization
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
        theta1 = 2*np.pi - theta1
        theta2 = 2*np.pi - theta2
        theta3 = 2*np.pi - theta3

        return np.rad2deg([theta1, theta2, theta3])    
    
    # Initial Position
    q_initial = endeff2joints(0.18,0.05,0)
    # Desired Position
    q_desired = endeff2joints(0.3,0.05,0)

    # Initial Joint Velocities
    qdot_initial = [0, 0, 0]
    # Desired Joint Velocities
    qdot_desired = [0, 0, 0]

    # Desired Joint Acceleration
    qddot_desired = [0, 0, 0]

    # Proportional Gain
    K_P = np.array([[100, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])

    # Derivative Gain
    K_D = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    
    # Integral Gain
    K_I = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])
    import sim  # Make sure sim.py is available

    # Connect to CoppeliaSim
    print("Connecting to CoppeliaSim...")
    sim.simxFinish(-1)  # Close any previous connections
    client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    print(client_id)

    time.sleep(1)

    if client_id == -1:
        print("❌ Failed to connect to CoppeliaSim.")
        sys.exit()
    else:
        print("✅ Connected to CoppeliaSim.")

        res, joint_names = sim.simxGetObjects(client_id, sim.sim_handle_all, sim.simx_opmode_blocking)
        print(res,joint_names)

        # Define joint names matching the CoppeliaSim scene
        #joint_names = ["Revolute_joint1", "Revolute_joint2", "Revolute_joint3"]

        # Start joint position streaming (important for velocity estimation later)
        for name in joint_names:
            err_code, handle = sim.simxGetObjectHandle(client_id, name, sim.simx_opmode_blocking)
            if err_code != sim.simx_return_ok:
                print(f"Failed to get handle for {name}")
            sim.simxGetJointPosition(client_id, handle, sim.simx_opmode_streaming)

        # Create simulated motor group object
        motor_group = SimulatedMotorGroup(joint_names, client_id)

        # Make controller
        controller = InverseDynamicsController(
            motor_group=motor_group,
            K_P=K_P,
            K_D=K_D,
            K_I=K_I,
            q_initial_deg=q_initial,
            q_desired_deg=q_desired,
            qdot_initial_deg_per_s=qdot_initial,
            qdot_desired_deg_per_s=qdot_desired,
            qddot_desired_deg_per_s2=qddot_desired
        )

        # Run controller
        controller.start_control_loop()
