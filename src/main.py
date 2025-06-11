import math
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sim  # CoppeliaSim remote API

l1, l2, l3 = 0.15, 0.1, 0.07

class InverseDynamicsController:
    def __init__(
        self,
        clientID: int,
        joint_handles: list,
        K_P: np.ndarray,
        K_D: np.ndarray,
        q_initial_deg: list[float],
        q_desired_deg: list[float],
        qdot_initial_deg_per_s: list[float],
        qdot_desired_deg_per_s: list[float],
        qddot_desired_deg_per_s2: list[float],
        max_duration_s: float = 8.0,
    ):
        # Save simulation-related handles
        self.clientID = clientID
        self.joint_handles = joint_handles

        # Save gains and control parameters
        self.K_P = np.asarray(K_P, dtype=np.double)
        self.K_D = np.asarray(K_D, dtype=np.double)
        self.q_initial_rad = np.deg2rad(q_initial_deg)
        self.q_desired_rad = np.deg2rad(q_desired_deg)
        self.qdot_initial_rad_per_s = np.deg2rad(qdot_initial_deg_per_s)
        self.qdot_desired_rad_per_s = np.deg2rad(qdot_desired_deg_per_s)
        self.qddot_desired_rad_per_s2 = np.deg2rad(qddot_desired_deg_per_s2)
        self.max_duration_s = max_duration_s

        # For plotting
        self.joint_position_history = []
        self.time_stamps = []

        # Link lengths (same as original)
        self.l1, self.l2, self.l3 = l1, l2, l3
    
        self.joint_position_history = []
        self.time_stamps = []

def start_control_loop(self):
    # Go to home configuration
    self.go_to_home_configuration()

    start_time = time.time()
    current_time = 0

    while current_time < self.max_duration_s:
        # Step 1: Get actual joint positions
        q_rad = []
        for handle in self.joint_handles:
            res, pos = sim.simxGetJointPosition(self.clientID, handle, sim.simx_opmode_blocking)
            q_rad.append(pos)
        q_rad = np.array(q_rad)

        # Save for plotting
        self.joint_position_history.append(q_rad)
        self.time_stamps.append(time.time() - start_time)

        # Step 2: Calculate position error
        q_error = self.q_desired_rad - q_rad

        # Use PD control for desired joint positions
        target_angles = self.q_desired_rad

        # Step 3: Send new target positions to CoppeliaSim
        for i, handle in enumerate(self.joint_handles):
            sim.simxSetJointTargetPosition(self.clientID, handle, target_angles[i], sim.simx_opmode_oneshot)

        # Wait a bit to let the simulation update
        time.sleep(0.05)  # 20 Hz loop
        current_time = time.time() - start_time

    # After loop, stop simulation
    sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot_wait)

def go_to_home_configuration(self):
    """Moves the simulated arm to the home position."""
    # Move each joint to its initial (home) angle
    for i, handle in enumerate(self.joint_handles):
        sim.simxSetJointTargetPosition(
            self.clientID,
            handle,
            self.q_initial_rad[i],
            sim.simx_opmode_oneshot
        )

    # Wait for a short period to let the simulator move the joints
    time.sleep(2)  # Adjust this if needed

def endeff2joints(x, y, tilt):

        tilt = np.deg2rad(tilt)
        
        x0 = x - l3*np.cos(tilt)
        y0 = y - l3*np.sin(tilt)
        
        theta1A = np.arctan(y0/x0)
        theta1B = np.arccos((l1**2 + x0**2 + y0**2 - l2**2)/(2*l1*np.sqrt(x0**2 + y0**2)))
        
        theta1 = np.pi + theta1A + theta1B
        theta2 = np.arccos((l2**2 + l1**2 - x0**2 - y0**2)/(2*l2*l1))
        theta3 = 2*np.pi - (theta1A + theta1B) - theta2 + tilt

        theta1 = 2*np.pi - theta1
        theta2 = 2*np.pi - theta2
        theta3 = 2*np.pi - theta3

        return np.rad2deg([theta1, theta2, theta3])
# Initial Position
q_initial = endeff2joints(0.3, 0.05, 0)
print("home position")
print(q_initial)

# Desired Position (for example)
q_desired = endeff2joints(0.3, 0.05, 0)

# Final Position (if used in your motion sequence)
q_final = endeff2joints(0.3, 0.05, 10)

# Initial, desired, and final joint velocities
qdot_initial = [0, 0, 0]
qdot_desired = [0, 0, 0]
qdot_final = [0, 0, 0]

# Desired Joint Acceleration
qddot_desired = [0, 0, 0]

#TODO Tune Gains
# Proportional Gain
K_P = np.array([[3, 0, 0],
                [0, 1.85, 0],
                [0, 0, 1.75]])

# Derivative Gain
K_D = np.array([[0.16, 0, 0],
                [0, 0.04, 0],
                [0, 0, 0.13]])

if __name__ == "__main__":
    # Connect to CoppeliaSim
    sim.simxFinish(-1)
    clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

    if clientID != -1:
        print("Connected to remote API server")
        sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot_wait)

        # Get joint handles
        joint_names = ['/Revolute_joint1', '/Revolute_joint2', '/Revolute_joint3']
        joint_handles = []
        for name in joint_names:
            res, handle = sim.simxGetObjectHandle(clientID, name, sim.simx_opmode_blocking)
            joint_handles.append(handle)

        # Set initial position (home)
        q_initial_rad = np.deg2rad(q_initial)
        for i, handle in enumerate(joint_handles):
            sim.simxSetJointTargetPosition(clientID, handle, q_initial_rad[i], sim.simx_opmode_oneshot)
        time.sleep(2)  # Let it move to home

        # Set desired position
        q_desired_rad = np.deg2rad(q_desired)

        # Start control loop
        start_time = time.time()
        current_time = 0
        joint_position_history = []
        time_stamps = []

        while current_time < 8.0:  # max duration
            # Get actual joint positions
            q_rad = []
            for handle in joint_handles:
                res, pos = sim.simxGetJointPosition(clientID, handle, sim.simx_opmode_blocking)
                q_rad.append(pos)
            q_rad = np.array(q_rad)

            # Save for plotting
            joint_position_history.append(q_rad)
            time_stamps.append(time.time() - start_time)

            # PD control: compute position error
            q_error = q_desired_rad - q_rad

            # For simulation, just send the desired angles directly
            target_angles = q_desired_rad

            # Send target angles to CoppeliaSim
            for i, handle in enumerate(joint_handles):
                sim.simxSetJointTargetPosition(clientID, handle, target_angles[i], sim.simx_opmode_oneshot)

            time.sleep(0.05)
            current_time = time.time() - start_time

        # Stop simulation
        sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot_wait)
        sim.simxFinish(clientID)
        print("Simulation ended")

        # Plotting
        # After control loop
        joint_history = np.array(joint_position_history).T  # shape: (3, N)
        time_stamps = np.array(time_stamps)

# ----------------------------------------------------------------------------------
# Plot Results
# ----------------------------------------------------------------------------------
        date_str = datetime.now().strftime("%d-%m_%H-%M-%S")
        fig_file_name = f"joint_positions_vs_time_{date_str}.pdf"

# Create figure and axes
        fig, (ax_motor0, ax_motor1, ax_motor2) = plt.subplots(3, 1, figsize=(10, 12))

# Label Plots
        fig.suptitle("Joint Angles vs Time")
        ax_motor0.set_title("Joint 1")
        ax_motor1.set_title("Joint 2")
        ax_motor2.set_title("Joint 3")

        ax_motor2.set_xlabel("Time [s]")
        ax_motor0.set_ylabel("Angle [deg]")
        ax_motor1.set_ylabel("Angle [deg]")
        ax_motor2.set_ylabel("Angle [deg]")

# Desired angles (if you want convergence bounds)
        ax_motor0.axhline(q_desired[0], ls="--", color="red", label="Setpoint")
        ax_motor1.axhline(q_desired[1], ls="--", color="red", label="Setpoint")
        ax_motor2.axhline(q_desired[2], ls="--", color="red", label="Setpoint")

        ax_motor0.axhline(q_desired[0] - 1, ls=":", color="blue", label="Convergence Bound")
        ax_motor0.axhline(q_desired[0] + 1, ls=":", color="blue")
        ax_motor1.axhline(q_desired[1] - 1, ls=":", color="blue", label="Convergence Bound")
        ax_motor1.axhline(q_desired[1] + 1, ls=":", color="blue")
        ax_motor2.axhline(q_desired[2] - 1, ls=":", color="blue", label="Convergence Bound")
        ax_motor2.axhline(q_desired[2] + 1, ls=":", color="blue")

        ax_motor0.axvline(1.5, ls=":", color="purple")
        ax_motor1.axvline(1.5, ls=":", color="purple")
        ax_motor2.axvline(1.5, ls=":", color="purple")

# Plot joint trajectories
        ax_motor0.plot(time_stamps, np.degrees(joint_history[0]), color="black", label="Joint 1 Trajectory")
        ax_motor1.plot(time_stamps, np.degrees(joint_history[1]), color="black", label="Joint 2 Trajectory")
        ax_motor2.plot(time_stamps, np.degrees(joint_history[2]), color="black", label="Joint 3 Trajectory")

        ax_motor0.legend()
        ax_motor1.legend()
        ax_motor2.legend()

        fig.savefig("motorplots.png")
        plt.show()

    else:
        print("Failed to connect to remote API server")
