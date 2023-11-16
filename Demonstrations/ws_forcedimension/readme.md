# Execute
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


# Acknowledegement
Refer to [ICube-Robotics/forcedimension_ros2](https://github.com/ICube-Robotics/forcedimension_ros2) for details.