# Pit Stop Robot 🚗🤖

Improving Speed and Safety in Fast-Paced Motorsport Environments
Overview

The Pit Stop Robot is a prototype robotic arm designed to automate the wheel-changing task in motorsport pit stops. Developed as a proof-of-concept for reducing pit crew risk and improving pit stop times, the robot focuses on quickly and accurately aligning with lug nuts for removal and reattachment. This project was created as part of UCLA's MAE C263C course on Robotic Systems Control.
Features

- 3DOF planar robotic arm with revolute joints
- Inverse dynamics controller with gravity and Coriolis compensation
- PID control with additional integral term for steady-state error
- Python-based control interface
- Simulated in CoppeliaSim with Python API integration
- Physical prototype built using FDM 3D-printed components and Dynamixel motors

System Architecture

- Language: Python
- Simulation: CoppeliaSim (Edu Edition via LegacyRemoteAPI)
- Hardware:
  - 4x Dynamixel MX-28AR motors
  - U2D2 USB communication converter
  - PLA 3D-printed linkages
- Control: Inverse Dynamics Controller with PID tuning
- Kinematics: Custom inverse kinematics solution for end-effector positioning

Demonstrated Tasks

The robot completes a five-phase motion cycle:
- Initial Position
- Approach Lug Nut (Loosen)
- Retract for Wheel Swap
- Approach Lug Nut (Tighten)
- Final Rest Position

Limitations

- Motor strength and model inaccuracies introduced steady-state and oscillation errors
- End-effector alignment drifted off-target during retraction phase
- Simulation control integration faced API limitations and debugging challenges

Future Work

- Replace inverse dynamics controller with robust control for better error tolerance
- Implement force control for torque-based lug nut manipulation
- Add computer vision and a gantry system for full automation and 3D alignment
- Upgrade to ZeroMQ API in CoppeliaSim for improved simulation fidelity
- Use stronger materials like aluminum or carbon fiber for improved rigidity
