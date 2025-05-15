# Project Introduction

A Semi-Supervised Learning Framework for Video Segmentation

This project provides tools for training and inference in video segmentation tasks, supporting applications like scene parsing and object segmentation.

# Model Structrue

![image](https://github.com/user-attachments/assets/87273e56-59f9-443d-b237-c3287c951920)

# Display of project results

![image](https://github.com/user-attachments/assets/39af8391-f240-4b59-8f3e-3ed6a89fd5dd)

# folder introduce

![image](https://github.com/user-attachments/assets/a92100e7-9f28-416a-a1cd-cea8321e9553)

📁 PyTorch-Correlation-extension
Custom CUDA-optimized correlation operations for dense correspondence tasks

📁 configs
YAML/JSON configuration files for model hyperparameters, dataset paths, and training/evaluation settings.

📁 dataloader
Data pipeline scripts for loading, preprocessing, and augmenting video datasets (supports multi-threaded loading).

📁 datasets
Raw and preprocessed video datasets (e.g., DAVIS, YouTube-VOS) with split files (train/val/test).

📁 davis2017-evaluation
Official evaluation scripts and metrics (e.g., Jaccard index, F-measure) for DAVIS 2017 benchmark compatibility.

📁 img_logs
Training visualization outputs: segmented frames, attention maps, and qualitative results (stored as images/GIFs).

📁 networks
Model architectures and pretrained weight loaders.

📁 pretrain_models
Downloaded pretrained models  for transfer learning or fine-tuning.

📁 tools**
Utility scripts for dataset conversion, annotation tools, or third-party library integrations.

📁 utils
Core helper functions: loss calculations, logging, progress bars, and GPU memory management.

📄 train_eval.shBash script to launch training/evaluation pipelines with configurable arguments (e.g., --gpu 0 --dataset davis).



# data acquisition

Due to the upper limit on the file size uploaded by GitHub, the dataset and model weights cannot be uploaded here.

pretrain_models: https://cowtransfer.com/s/d2546440518240

Davis_medaka_dataset: https://cowtransfer.com/s/3d0ed4b3d8ae4b

DAVIS-2016-trainval:https://davischallenge.org.

DAVIS-2017-trainval-480p:https://davischallenge.org.



# How to use:

1, Download the required dataset and project code
2, configuration environment
Deep learning environment：conda install pytorch==2.3.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch=3.12
Other libraries: Install whatever is missing.
3, Ensure the relative position of files
If not correct, follow the code prompts to make corrections
4, run sh train_eval.sh
