# ALL_TASKS = ["lift", "can", "square", "transport", "tool_hang", "lift_real", "can_real", "tool_hang_real"]
# ALL_DATASET_TYPES = ["ph", "mh", "mg", "paired"]
# ALL_HDF5_TYPES = ["raw", "low_dim", "image", "low_dim_sparse", "low_dim_dense", "image_sparse", "image_dense"]



# raw
python /root/RoboLearn/Packages/robomimic-1.3/robomimic/scripts/download_datasets.py --download_dir /root/RoboLearn/Download/Data --tasks all --dataset_types all --hdf5_types raw --dry_run

task: lift                                  dataset type: ph
task: lift                                  dataset type: mh
task: lift                                  dataset type: mg
task: can                                   dataset type: ph
task: can                                   dataset type: mh
task: can                                   dataset type: mg
task: can                                   dataset type: paired
task: square                                dataset type: ph
task: square                                dataset type: mh
task: transport                             dataset type: ph
task: transport                             dataset type: mh
task: tool_hang                             dataset type: ph
task: lift_real                             dataset type: ph
task: can_real                              dataset type: ph
task: tool_hang_real                        dataset type: ph

python /root/RoboLearn/Packages/robomimic-1.3/robomimic/scripts/download_datasets.py --download_dir /root/RoboLearn/Download/Data --tasks tool_hang_real --dataset_types all --hdf5_types raw

# low_dim
python /root/RoboLearn/Packages/robomimic-1.3/robomimic/scripts/download_datasets.py --download_dir /root/RoboLearn/Download/Data --tasks all --dataset_types all --hdf5_types low_dim --dry_run

task: lift                                  dataset type: ph
task: lift                                  dataset type: mh
task: can                                   dataset type: ph
task: can                                   dataset type: mh
task: can                                   dataset type: paired
task: square                                dataset type: ph
task: square                                dataset type: mh
task: transport                             dataset type: ph
task: transport                             dataset type: mh
task: tool_hang                             dataset type: ph


# image
python /root/RoboLearn/Packages/robomimic-1.3/robomimic/scripts/download_datasets.py --download_dir /root/RoboLearn/Download/Data --tasks all --dataset_types all --hdf5_types image --dry_run

task: lift                                  dataset type: ph
task: lift                                  dataset type: mh
task: can                                   dataset type: ph
task: can                                   dataset type: mh
task: can                                   dataset type: paired
task: square                                dataset type: ph
task: square                                dataset type: mh
task: transport                             dataset type: ph
task: transport                             dataset type: mh
task: tool_hang                             dataset type: ph


# low_dim_sparse
python /root/RoboLearn/Packages/robomimic-1.3/robomimic/scripts/download_datasets.py --download_dir /root/RoboLearn/Download/Data --tasks all --dataset_types all --hdf5_types low_dim_sparse --dry_run

task: lift                                  dataset type: mg
task: can                                   dataset type: mg


# low_dim_dense
python /root/RoboLearn/Packages/robomimic-1.3/robomimic/scripts/download_datasets.py --download_dir /root/RoboLearn/Download/Data --tasks all --dataset_types all --hdf5_types low_dim_dense --dry_run

task: lift                                  dataset type: mg
task: can                                   dataset type: mg
