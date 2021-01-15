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
In a new terminal
```sh
$ roslaunch realsense2_camera rs_camera.launch 
```

In a new terminal, launch ar track module by:
```sh
roslaunch ar_track_alvar pr2_indiv.launch marker_size:=5.55
```

In a new terminal, launch ar marker target processing node, it publishes laser and servo command.
```sh
$ export ROS_HOSTNAME=192.168.0.101
$ export ROS_IP=192.168.0.101
$ rosrun rr_control_input_manager ar_track_laser.py
```

