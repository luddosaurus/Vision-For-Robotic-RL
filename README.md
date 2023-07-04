# Vision for Robotic RL

**Setup**
- Robot arm (Franka Emika Panda)
- Camera (Realsense d435)
- ROS Noetic
- [OpenCv v4.6](https://docs.opencv.org/4.6.0/)
- Python v3.something
- ArUco / ChArUco

# Goal
## Camera Calibration
The main goal of this master thesis is to align a real world environment with a simulated one by using camera calibration. This alignment is crucial for the successful deployment of reinforcement learning (RL) models in real-world scenarios. In this research, we will investigate several research questions to achieve this goal.

- How can a real world setup best be aligned with a simulation?
- How can the quality of alignment be measured?
- What would be a suitable method for positioning cameras?
- How can the alignment be done such that the RGB (Red, Green, Blue) and depth (distance to object) images are aligned with each other, both in reality and simulation.
- Can evaluation be done equally well for ArUco:s and detected objects?

## Reinforcement Learning
The primary aim of camera alignment is to enhance system performance. Al-though sim2real transfer learning is a potential area of application for the aligned camera, its effectiveness is subject to high levels of non-determinism, making it a secondary focus of this master’s thesis. As such, any success achieved in this area will be considered a valuable bonus outcome.

- Would an aligned simulation and real setup facilitate sim2real transferlearning in robotics?


# Use
check `docs/`
