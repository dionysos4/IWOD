# A Deep Network for Object Detection on Inland Waters

The code will be available soon.


## KITTI Dataset Preparation (Multiview & Unrectified)

The standard KITTI Object Detection dataset consists of stereo-rectified images. However, to demonstrate that our approach is capable of handling **unrectified stereo data** and supports **multiview inputs**, we generate our dataset directly from the KITTI Raw recordings. This allows us to include the grayscale camera streams (Cam 0 & Cam 1), which are absent from the official detection benchmark.

Please follow the steps below to prepare the data.

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

### 3. Add Labels and Pointclouds
Finally, you must manually download the official labels and LiDAR data to complete the dataset structure:

1. Go to the [KITTI 3D Object Detection Evaluation Page](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
2. Download the following:
   * **Training labels** of object data set (5 MB)
   * **Velodyne point clouds** (29 GB)
3. Extract and place these files into your `multiview_kitti` folder.



# Installation
conda create --name iwod python==3.11
conda activate iwod
conda install -c conda-forge uv
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

cd iou_utils/cuda_op
(evtl. export TORCH_CUDA_ARCH_LIST="8.6")
uv pip install --no-build-isolation .
uv pip install pytorch-lightning