

# Dependencies
- ROS2(galactic)
- robomimi-1.3
- robosuite-offline
- stable-baselines3

We provide a docker image [xizobu/galactic:3.0](https://hub.docker.com/repository/docker/xizobu/galactic/general)
# Result
## Different Noise Angles and Datasets
- Effect of Noise Angle.  
&nbsp;&nbsp;IQ-Learn: baseline algorithm;  
&nbsp;&nbsp;IQ-Learn (filter): Just filtering noise without using confidence, it becomes IQ-Learn when θn is set to 180°;  
&nbsp;&nbsp;CIQL-E: Just filtering noise and using confidence;  
&nbsp;&nbsp;CIQL-A: Penalizing noise and using confidence.  
&nbsp;&nbsp;Ranking of algorithm performance: CIQL-A (40.3%) > CIQL-E (30.1%) > CIQL (filter, 26.8%) > IQ-Learn. 
&nbsp;&nbsp;Compared to simply filtering noise, implementing fine-grained confidence assessment on the demonstration data can effectively &nbsp;&nbsp;enhance the performance of the algorithm. Additionally, penalizing noise is also superior to straightforward noise filtering.

<div align=center>
<img src="https://github.com/XizoB/CIQL/blob/main/Confidence-based-IQ-Learn/results/Boundary%20Angle%20Evaluation.png" width="600" height="335">
</div>

## CIQL Evaluation
- Recovering environment rewards.  
&nbsp;&nbsp;Reward function recovered by CIQL-A aligns more closely with human intent.  
&nbsp;&nbsp;Evaluation and penalization of noise in the data are more aligned with human intentions compared to strategies trained with simple &nbsp;&nbsp;noise filtering.

<div align=center>
<img src="https://github.com/XizoB/CIQL/blob/main/Confidence-based-IQ-Learn/results/CIQL%20Evaluation.png" width="800" height="250">
</div>

- Performance of CIQLs and IQ-Learns  
&nbsp;&nbsp;IQ-Learn: success rate of the task is very low;  
&nbsp;&nbsp;IQ-Learn(filter): there are multiple cases where the robotic arm flies directly in a messy manner;  
&nbsp;&nbsp;CIQL-Expert: the decision time is long and the grasping is not decisive enough;   
&nbsp;&nbsp;CIQL-Agent: the decision time is short and the grasping is decisive.
<div align=center>
<img src="https://github.com/XizoB/CIQL/blob/main/Confidence-based-IQ-Learn/results/IQ-Learn%20.gif" width="350" height="300"> <img src="https://github.com/XizoB/CIQL/blob/main/Confidence-based-IQ-Learn/results/IQ-Learn(filter)%20.gif" width="350" height="300">
  
**(a)Performance of IQ-Learn** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **(b)Performance of IQ-Learn(filter)** 

<img src="https://github.com/XizoB/CIQL/blob/main/Confidence-based-IQ-Learn/results/CIQL-Expert.gif" width="350" height="300"> <img src="https://github.com/XizoB/CIQL/blob/main/Confidence-based-IQ-Learn/results/CIQL-Agent.gif" width="350" height="300">

**(c)Performance of CIQL-E** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **(d)Performance of CIQL-A** 
        
</div>

## Demonstrations Evaluation
- Noise filtering visualization of two human datasets, better and worse.  
&nbsp;&nbsp;After filtering out the cluttered trajectories, an organized trend emerges.  
&nbsp;&nbsp;Fine-grained confidence scores can be provided for human demonstration data without the need for active supervision signals from &nbsp;&nbsp;humans, a true reward function from the environment, or strict assumptions about noise.

<div align=center>
<img src="https://github.com/XizoB/CIQL/blob/main/Confidence-based-IQ-Learn/results/Demonstrations%20Evaluation.png" width="750" height="400">
</div>

# Run
## Collect demonstartions
1. Activate Omgea.x device
- Compile by using  
`colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --symlink-install`  
Note that using the local python environment rather than the conda
- Initialize Omega.x by running `./HapticDesk` in a terminal under the file path, Demonstrations/ws_forcedimension/src/forcedimension_ros2/fd_hardware/external/fd_sdk/bin

- Open two terminals on ws_forcedimension files and source the workspace  
`source install/setup.bash`  

- Running the driver in one terminal  
`ros2 launch fd_bringup fd.launch.py`  

- Publish end position data in another terminal  
`ros2 run tcp_socket ee_topic_sub`
2. Start a demonstration task  
refer to /root/RoboLearn/Demonstrations/launch/run.sh  
`python collect_human_demonstrations.py --robots IIWA --environment Lift --device omega` 

3. Merge demonstrations (demo = demo1 + demo2 ...)  
`python demonstration_merge.py --merge_directory collect_demonstration/Lift/IIWA_OSC_POSE`

4. Converted data （demo -> pkl）  
`python demonstration_transition.py --dataset_type robosuite_demo.hdf5 --output_dir iqlearn_demonstrations --dataset_path Lift/IIWA_OSC_POSE`


## Train and Evalute Agent
5. Train CIQL Agent
Refer to /root/RoboLearn/Confidence-based-IQ-Learn/run_confidence.sh  
IQ-Learn(IQ), CIQL-A(max_lamb) and CIQL-E(conf_expert)  
`python train_iq_dyrank.py env=robosuite_Lift_IIWA env.demo=robosuite_Lift_IIWA_better_worse_failed_90.pkl agent=sac agent.actor_lr=5e-06 agent.critic_lr=5e-06 agent.init_temp=0.001 expert.demos=90 seed=1 train.boundary_angle=30 C_aware.conf_learn=max_lamb`

6. Evalute CIQL Agent  
`python test_iq_dyrank.py env=robosuite_Lift_IIWA agent=sac env.has_renderer=False eval.policy=xxx`

7. Evalute Demonstrations using reward function recovered by CIQL  
`python test_iq_reward.py env=robosuite_Lift_IIWA env.demo=robosuite_Lift_IIWA_50.pkl expert.demos=50 agent=sac eval.policy=xxx`


# Acknowledegement
Thanks to the authors of [IQ-Learn](https://github.com/Div99/IQ-Learn) for their work and sharing!  
The code structure is based on the repo [IQ-Learn](https://github.com/Div99/IQ-Learn).  
Details of the Omega device driver with its ROS2 workspace can be found in the [ICube-Robotics/forcedimension_ros2](https://github.com/ICube-Robotics/forcedimension_ros2).
