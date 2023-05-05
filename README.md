# Vision for Robotic RL

**Setup**
- Robot arm (Franka Emika Panda)
- Camera (Realsense d435)
- ROS Noetic
- OpenCv v4.6?
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

## Camera Calibration
Code @ `code/catkin_ws/src/camera_calibration`



### Find the instrinsic camera parameters

> Finds the intrinsic parameters of the camera to compensate for distortion

0. Take a ChArUco board
1. Run
```bash
roslaunch camera_calibration internal.launch 
```

2. Take a picture of board
3. Move ChArUco board to a different angle/position
4. Repeat (2-3) until calibrated

**Intrinsic camera matrix** found in `data/intrinsic_camera_calibration`



### Find pose of the camera

> Finds the pose of the camera in relation to the robot-base

0. Attach ChArUco to the robot arms end-effector
1. Run
```bash
roslaunch camera_calibration external.launch
```

2. Take a picture
3. Move arm
4. Repeat (2-3) until calibrated

**Calibration data** stored in `data/calibration_transforms`

**Estimated Camera Position** stored in `data/camera_position` as a json of a StampedTransform
```json
[
  {
    "time": 1683021102.6665049,
    "frame_id": "base",
    "child_frame_id": "camera",
    "translation": {
      "x": 0.9178118955407216,
      "y": 0.12535867091010136,
      "z": 0.06776630208315132
    },
    "rotation": {
      "x": -0.38779971087219883,
      "y": -0.49610628481181585,
      "z": 0.5983656050652284,
      "w": 0.4954276345669261
    }
  }
]

```


## Simulation
???

