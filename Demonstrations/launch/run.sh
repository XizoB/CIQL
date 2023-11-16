"""
env_name:
        Lift, Stack, StackObstacle, NutAssembly, NutAssemblySingle, NutAssemblySquare, 
        NutAssemblyRound, PickPlace, PickPlaceSingle, PickPlaceMilk, 
        PickPlaceBread, PickPlaceCereal, PickPlaceCan, Door, 
        Wipe, ToolHang, TwoArmLift, TwoArmPegInHole, 
        TwoArmHandover, TwoArmTransport

robots:
        Panda, Sawyer, IIWA, Jaco, Kinova3, UR5e, Baxter

render_camera:
        frontview, birdview, agentview, sideview,
        robot0_robotviewros2 run tcp_socket ee_topic_sub, robot0_eye_in_hand

gripper_types:
        Panda   ---  PandaGripper
        Sawyer  ---  RethinkGripper
        IIWA    ---  Robotiq140Gripper
        Jaco    ---  JacoThreeFingerGripper
        Kinova3 ---  Robotiq85Gripper
        UR5e    ---  Robotiq85Gripper
        Baxter  ---  RethinkGripper

controller_configs:
        JOINT_POSITION, JOINT_TORQUE, JOINT_VELOCITY, OSC_POSITION, OSC_POSE, IK_POSE

"""

##############################################################  收集数据
#--------- 收集演示数据 collect_human_demonstration ----------------------#
python /root/RoboLearn/Demonstrations/launch/collect_human_demonstrations.py --directory /root/RoboLearn/Demonstrations/launch/collect_demonstration --robots IIWA --environment Lift --device omega


#--------- 演示数据复现仿真(可添加噪声)与收集 expert_generation ----------------------# 
python /root/RoboLearn/Demonstrations/launch/robomimic_expert_generation.py --folder /root/RoboLearn/Demonstrations/launch/collect_demonstration/Lift


#--------- 演示数据数值仿真(查看演示数据有无异常值) numerical_simulation ----------------------# 
python /root/RoboLearn/Demonstrations/launch/numerical_simulation.py --folder /root/RoboLearn/Demonstrations/launch/collect_demonstration/Lift/IIWA_OSC_POSE_FAILED_10


##############################################################  数据处理
#--------- 合并演示数据  ----------------------#
# demo = demo1 + demo2 ...... 
python /root/RoboLearn/Demonstrations/launch/scripts/demonstration_merge.py --merge_directory /root/RoboLearn/Demonstrations/launch/collect_demonstration/Lift/IIWA_OSC_POSE


#--------- 演示数据转换 demonstration_transition ----------------------#
# demo -> pkl
python /root/RoboLearn/Demonstrations/launch/scripts/demonstration_transition.py --dataset_type robosuite_demo.hdf5 --output_dir /root/RoboLearn/Demonstrations/launch/iqlearn_demonstrations --dataset_path /root/RoboLearn/Test/Test_launch/collect_demonstration/Lift/IIWA_OSC_POSE


#--------- 合并pkl mix_pkl ----------------------#
# pkl = pkl1 + pkl2
python /root/RoboLearn/Demonstrations/launch/scripts/mix_pkl.py --folder /root/RoboLearn/Demonstrations/launch/iqlearn_demonstrations/Lift/IIWA/better_worse_failed_90 --env-id robosuite_Lift_IIWA_better_worse_failed_90




##############################################################  可视化
#--------- 演示数据实时重现 playback_demonstrations_from_hdf5 ----------------------# 
python /root/RoboLearn/Demonstrations/launch/playback_demonstrations_from_hdf5.py --use-actions True --folder /root/RoboLearn/Demonstrations/launch/collect_demonstration/Lift/IIWA_OSC_POSE_optimal/raw/OSC_POSE_93_29.669



#--------- 演示数据全功能重现 Robomimic playback_dataset  ----------------------#
# 对于前 n 个轨迹，逐一加载环境模拟器状态，并渲染 "agentview" 和 "robot0_eye_in_hand" 摄像机到/tmp/playback_dataset.mp4的视频
python /root/RoboLearn/Demonstrations/launch/playback_dataset.py --render_image_names frontview robot0_eye_in_hand robot0_robotview --dataset /root/RoboLearn/Download/Data/can/ph/demo.hdf5 --video_path /root/RoboLearn/Demonstrations/launch/playback_demonstration/can.mp4 --n 5


#--------- 可视化各个演示数据的初始化状态 ----------------------#
python /root/RoboLearn/Demonstrations/launch/playback_dataset.py --render_image_names agentview --dataset /root/RoboLearn/Download/Data/can/mg/demo.hdf5 --video_path /root/RoboLearn/Demonstrations/launch/playback_demonstration/obs_trajectory_can_4.mp4 --first --n 5
Demonstrations/launch/Data_train/iqlearn_demonstrations Demonstrations/launch/Data_train/collect_demonstration Demonstrations/launch/Data_train/playback_demonstration