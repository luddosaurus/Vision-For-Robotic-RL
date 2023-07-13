
Launch file : `code/catkin_ws/src/object_finder/launch/hsv_cubes.launch`

Script : `code/catkin_ws/src/object_finder/nodes/hsv_cubes_finder.py`

![Cube Finding](images/cube_finding.png)


## Settings
---

### Broadcast Camera Position
Add camera position estimate gathered from 
```
code/catkin_ws/src/camera_calibration/calibration_resuslts/eye_to_hand/cam_top/estimates.json
``` 
to the camera position broadcaster in :
```
code/catkin_ws/src/camera_estimate_broadcaster/camera_transforms/my_cameras.json
```

Example of camera position in broadcaster
```json
{  
	"CAM_TOP": {
	    "frame_id": "world",
	    "child_frame_id": "cam_top",
	    "translation": {
	      "x": 0.40758058215725995,
	      "y": 0.15088598354369197,
	      "z": 1.2137956007895734
	    },
	    "rotation": {
	      "x": -0.6881759456811474,
	      "y": 0.7252731707731805,
	      "z": -0.012537449930996898,
	      "w": -0.015346266376654994
	    },
},
	...
}
```


### Object Finder

**Launch Config**
Change camera stream topic
```xml
<arg name="camera_topic" default="/cam_top/color/image_raw"/>
```

**Things to change in the code**
1. Parent name for the cube
line 194 in `hsv_cubes_finder.py` 
```python
TFPublish.publish_static_transform(
    publisher=self.center_broadcaster,  
	parent_name='cam_top',
	child_name=f'cube',  
	rotation=[0., 0., 0., 1.],  
	translation=self.position
)
```

2. Camera Depth Topic
line 52 in `hsv_cubes_finder.py` 
```python:
self.aligned_depth_subscriber = rospy.Subscriber(									'/cam_top/aligned_depth_to_color/image_raw', 
	Image,  
	self.camera_depth_aligned_callback
)
```

## Start Everything
---
1. Start Arm
2. Start Camera
3. Start Object Finder
```
roslaunch object_finder hsv_cubes.launch
```


## Find a Cube
---

**Controls Overview**
```
u = pick up target
d = put down target
m = pick up target and move to random location
q = quit
o/p = scale window up/down
k/l = scale roi for color picking
```

**Sliders Overview**
```
hue = color spectrum 
value = color brightness
saturation = gray to colorful
fill = fill holes in the segment
noise = remove small segments
```


**Find Cube**
1. Click on the colored cube to pick up
2. Change sliders to only segment the target

**Moving the Cube**
Press `[u]` for pick up target
Press `[d]` for put down target
Press `[m]` for random put down target