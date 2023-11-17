

# Execute
## Activate Omgea.x device
1. Compile by using  
`colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --symlink-install`  
Note that using the local python environment rather than the conda
2. Initialize Omega.x by running `./HapticDesk` in terminal
Demonstrations/ws_forcedimension/src/forcedimension_ros2/fd_hardware/external/fd_sdk/bin

3. Open two terminals on ws_forcedimension files and source the workspace  
`source install/setup.bash`  

4. Running the driver in one terminal  
`ros2 launch fd_bringup fd.launch.py`  

5. Publish end position data in another terminal
`ros2 run tcp_socket ee_topic_sub`
## Start a demonstration task
`python collect_human_demonstrations.py --directory collect_demonstration --robots IIWA --environment Lift --device omega`  
refer to /root/RoboLearn/Demonstrations/launch/run.sh

# Acknowledegement
Thanks to the authors of [IQ-Learn](https://github.com/Div99/IQ-Learn) for their work and sharing!  
The code structure is based on the repo [IQ-Learn](https://github.com/Div99/IQ-Learn).  
Details of the Omega device driver with its Ros2 workspace can be found in the [ICube-Robotics/forcedimension_ros2](https://github.com/ICube-Robotics/forcedimension_ros2).