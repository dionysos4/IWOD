# A Deep Network for Object Detection on Inland Waters

## Lake Constance Obstacle Detection Dataset Preparation
Follow these steps to set up the Lake Constance Obstacle Detection (LCOD) dataset and initiate the training pipeline.

### Donwload and Extract Dataset
1. Download the [Lake Constance Obstacle Detection Dataset](https://kondata.uni-konstanz.de/radar/de/dataset/FyOPnLSQDKwxfbdO) and extract the files.

2. The raw HDF5 files must be converted into a KITTI-style format for compatibility with the training pipeline. Use the **hdf5_extractor.py** script provided in the **lcod_tools** directory
```bash
cd lcod_tools
python hdf5_extractor.py --dataset_dir="<path to Lake_Constance_Obstacle_Detection_Data-set>" 
                        --save_dir="<path where you want to save it>" 
                        --downsample_factor=<factor to downnsample image size (e.g. 2 is 1/2 image size)> 
                        --split_file_path <absolute path to ../config/lcod_config>
```

## Inland Water Training

To train the Lake Constance model, choose and customize a configuration file from the `config` folder.
Then, provide the path to this configuration file and run:
```bash
python train_lcod.py --config_path <path/to/your/config.yaml>
```


## KITTI Dataset Preparation (Multiview & Unrectified)

The standard KITTI Object Detection dataset consists of stereo-rectified images. However, to demonstrate that our approach is capable of handling **unrectified stereo data** and supports **multiview inputs**, we generate our dataset directly from the KITTI Raw recordings. This allows us to include the grayscale camera streams (Cam 0 & Cam 1), which are absent from the official detection benchmark.

Follow the steps below to prepare the data.

### 1. Download Raw Data
First, download both the synchronized and unsynchronized raw data sequences using the provided helper scripts:

```bash
cd kitti_tools/kitti_raw_sync
./raw_data_downloader.sh

cd ../kitti_raw_unsynced
./raw_data_downloader.sh
```
### 2. Extract and Map Dataset
We map the raw data sequences to the frames used in the official detection benchmark. This step also extracts the grayscale camera images (Cam 0 and Cam 1), which are typically omitted from the standard detection set.

* Run the iPython notebook: `create_dataset.ipynb`
* The processed data will be exported to the `multiview_kitti` directory.

### 3. Add Labels and Point Clouds
Finally, you must manually download the official labels and LiDAR data to complete the dataset structure:

1. Go to the [KITTI 3D Object Detection Evaluation Page](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
2. Download the following:
   * **Training labels** of object data set (5 MB)
   * **Velodyne point clouds** (29 GB)
3. Extract and place these files into your `multiview_kitti` folder.


## KITTI Training

To train the KITTI models, choose and customize a configuration file from the `config` folder.  
Then, provide the path to this configuration file and run:
```bash
python train_kitti.py --config_path <path/to/your/config.yaml>
```

## Installation
```bash
conda create --name iwod python==3.11
conda activate iwod
conda install -c conda-forge uv
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
conda install anaconda::cudatoolkit==11.8.0
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11
cd iou_utils/cuda_op
uv pip install --no-build-isolation .
uv pip install pytorch-lightning lightning scipy matplotlib tensorboard fire open3d opencv-python jupyterlab scikit-image numba
uv pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git
```