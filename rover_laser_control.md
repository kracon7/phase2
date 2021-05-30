# Control the laser on the rover by ROS
## launch the rover

```sh
$ sudo chmod 777 /dev/ttyUSB0
$ roslaunch rr_openrover_basic example.launch 
```
In the second terminal
```sh
$ roslaunch rr_control_input_manager example.launch 
```

In the third terminal
```sh
sudo xboxdrv
```
## launch the raspberry pi ROS node
In the fourth terminal
```sh
$ export ROS_HOSTNAME=192.168.0.101
$ export ROS_IP=192.168.0.101
$ ssh ubuntu@192.168.0.100
ubuntu@ubuntu:~$ rosrun laser_control laser_control.py
```

## launch realsense and ar track module


In a new terminal, launch realsense camera node

single camera:```$ roslaunch realsense2_camera rs_camera.launch ```

double cameras (cam1 front cam2 side): ```$ roslaunch realsense2_camera rs_multiple_devices.launch serial_no_camera1:=817412070531 serial_no_camera2:=817512070887 ```

In a new terminal, launch ar track module. 

single camera:
This is a customized launch file. The file can be found [here](https://drive.google.com/file/d/1miq6X2cE-JwfbShJRTCDqtUExPz_c5U9/view?usp=sharing)
```sh
roslaunch ar_track_alvar pr2_indiv.launch marker_size:=5.55
```

double cameras:
[file link](https://drive.google.com/file/d/1GQ4_lW7yanbjQKli5eGf49y-BIxuhWt7/view?usp=sharing)
```sh
roslaunch ar_track_alvar pr2_multi_camera.launch marker_size:=5.55
```

In a new terminal, launch ar marker target processing node, it publishes laser and servo command.
```sh
$ export ROS_HOSTNAME=192.168.0.101
$ export ROS_IP=192.168.0.101
$ rosrun rr_control_input_manager ar_track_laser.py
```

## Data collection using this setup

Launch the joystick message publish ROS node and then,

```
rosrun rr_control_input_manager joystick_camera_ctrl.py
```
