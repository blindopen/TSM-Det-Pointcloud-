# Transferable Class Statistics and Multi-scale Feature Approximation for 3D Object Detection

## Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 20.04)
* Python 3.10
* PyTorch 2.0
* CUDA 11.8
* NVIDIA 2080Ti 11G
* [`spconv v2.x`](https://github.com/traveller59/spconv)

The requirement.txt file is generated through pipreq. Actually, we found many other libraries in my environment, as shown in this video [Environment Video Baidu Netdisk](https://pan.baidu.com/s/1gjsPknpqFZpHoAJ5h6VS6g?pwd=7g28).

## Installation
a. Install the dependent libraries as mentioned above. **Please make sure that the environment is configured successfully**.
 
b. Install this `pcdet` library and its dependent libraries by running the following command, as shown in this video [Setup Video Baidu Netdisk](https://pan.baidu.com/s/1BJFyXH9I5eSYpGxsR2cXsQ?pwd=hb7x):
```shell
python setup.py develop
```

## Dataset Preparation

Currently we provide the dataloader of KITTI, Waymo. 

### KITTI Dataset
* Please download the official KITTI 3D object detection dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):


```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
### Waymo Open Dataset
* Please download the official Waymo Open Dataset, 
including the training data `training_0000.tar~training_0031.tar` and the validation 
data `validation_0000.tar~validation_0007.tar`.
* Unzip all the above `xxxx.tar` files to the directory of `data/waymo/raw_data` as follows (You could get 798 *train* tfrecord and 202 *val* tfrecord ):  
```
OpenPCDet
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_v0_5_0
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/  (old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl  (old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional, old, for single-frame)
│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)
|   |   |── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0 (new, for single/multi-frame)
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1_multiframe_-4_to_0.pkl (new, for single/multi-frame)
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0_global.np  (new, for single/multi-frame)
 
├── pcdet
├── tools
```
* Install the official `waymo-open-dataset` by running the following command: 
```shell script
pip3 install --upgrade pip
# tf 2.0.0
pip3 install waymo-open-dataset-tf-2-5-0 --user
```

* Extract point cloud data from tfrecord and generate data infos by running the following command (it takes several hours, 
and you could refer to `data/waymo/waymo_processed_data_v0_5_0` to see how many records that have been processed): 
```python 
# only for single-frame setting
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```

Note that you do not need to install `waymo-open-dataset` if you have already processed the data before and do not need to evaluate with official Waymo Metrics. 

## Training & Testing

### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```


### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
  
* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
```
## Result

### KITTI
The results are the 3D detection performance on the *val* set of KITTI dataset. [Pretrained model for KITTI ~5h Baidu Netdisk](https://pan.baidu.com/s/1Nz0FObDQFVo0Wm2YVkhH9Q?pwd=8mdp)
* Test with a pretrained model, as shown in the video. [test_kitti_video Baidu Netdisk](https://pan.baidu.com/s/1ml9H0JyEAmB6HRbgo1lPUA?pwd=23ts) 
```shell script
python test.py --cfg_file ${path_to_cfg_file_for_kitti} --batch_size ${BATCH_SIZE} --ckpt ${path_to_ckpt_for_kitti}
```
| Easy Car |Mod. Car |Hard Car | Easy Ped |Mod. Ped |Hard Ped | Easy Cyc | Mod. Cyc | Hard Cyc | 
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| 92.73|  85.62|  82.99|  63.03|  57.92|  52.26|  91.63|  72.13|  67.48| 

Actually, deep learning involves a certain amount of randomness, so we provide logs to prove the source of the experimental data.
```shell script
./res_kitti.txt
```

### Waymo Open Dataset
The model is trained with **a single frame** of **20% data (~32k frames)** of all the training samples, and the results of each cell here are mAP/mAPH on the **whole** validation set (version 1.2). [Pretrained model for Waymo~12h Baidu Netdisk](https://pan.baidu.com/s/1R2_jE-ADWclzuqUVkouGrQ?pwd=uvvb)    
* Test with a pretrained model, as shown in the video. [test_waymo_video Baidu Netdisk](https://pan.baidu.com/s/1KxkfFDsfwwH7V4FBCNPjCw?pwd=tnyh) 
```shell script
python test.py --cfg_file ${path_to_cfg_file_for_waymo} --batch_size ${BATCH_SIZE} --ckpt ${path_to_ckpt_for_waymo}
```
| Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |  
|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| 71.29/70.68|62.67/62.13|63.14/53.73	|55.10/46.74|	64.71/62.61 |	62.24 /60.22 | 


We also provide logs to prove the source of the experimental data.
```shell script
./res_waymo.txt
```
