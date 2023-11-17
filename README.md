

# Dependencies
# Execute
## Collect demonstartions
### Activate Omgea.x device
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
### Start a demonstration task
`python collect_human_demonstrations.py --robots IIWA --environment Lift --device omega`  
refer to /root/RoboLearn/Demonstrations/launch/run.sh

## Train CIQL Agent
`python train_iq_dyrank.py env=robosuite_Lift_IIWA env.demo=robosuite_Lift_IIWA_better_worse_failed_90.pkl agent=sac agent.actor_lr=5e-06 agent.critic_lr=5e-06 agent.init_temp=0.001 expert.demos=90 seed=1 train.boundary_angle=30 C_aware.conf_learn=IQ`
refer to /root/RoboLearn/Confidence-based-IQ-Learn/run_confidence.sh


# Acknowledegement
Thanks to the authors of [IQ-Learn](https://github.com/Div99/IQ-Learn) for their work and sharing!  
The code structure is based on the repo [IQ-Learn](https://github.com/Div99/IQ-Learn).  
Details of the Omega device driver with its ROS2 workspace can be found in the [ICube-Robotics/forcedimension_ros2](https://github.com/ICube-Robotics/forcedimension_ros2).