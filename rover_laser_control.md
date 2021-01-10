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

In the fourth terminal
```sh
$ export ROS_HOSTNAME=192.168.0.101
$ export ROS_IP=192.168.0.101
$ ssh ubuntu@192.168.0.100
ubuntu@ubuntu:~$ rosrun laser_control laser_control.py
```

In the fifth terminal
```sh
$ export ROS_HOSTNAME=192.168.0.101
$ export ROS_IP=192.168.0.101
$ rosrun rr_control_input_manager rover_laser_publisher.py --visuale 1 --save_img 0
```
