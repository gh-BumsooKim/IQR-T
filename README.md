# IQR-T
Irregular Quadrilateral Region Tracking using ARIMA for Dynamic Projection Mapping

![image](https://user-images.githubusercontent.com/67869508/169051499-9072b3ca-3b18-45ec-a5da-bafc9c2cc5ec.png)

## Requirements

* python == 3.8.12
* numpy == 
* opencv-python == 


Recommend to used python virtual environment like anaconda.

```
pip install -r requirements.txt

(recommended)
conda install conda.yml
```

## Usage

```cmd
python.exe main.py 
```

* `--cont_prefix` : path with prefix of content images
* `--calibration_config` : path of the file containing the camera-projector calibration parameters

(if you want to see more information about obtaining camera-calibration parameters, [click here]() )
* `--track_mode` : mode about region tracking as shown in the following table.

| track_mode | Method | E |
|:---:|:---:|:---:|
| 0 (default) | AutoRegression |

(Optional Flags)
* `--explain` : 

# Exhibition Installation Guide

## A. Prepare the RGBD Camera and Projector

You can use the any RGBD camera that can export the depth map, but in this project, we used [MS Azure Kinect]().
Similarly, you can use the any projector that is suppported the 1280 × 720 resolution image.

## B. Fix the Camera and Projector Phsically and Organize Exhibition for Dynamic Projection Mapping

## C. Obtain RGBD Camera-Projector Calibration Parameters

In this process, camera-view and projector-view are corresponed with same scale and ratio. You can use our auto calibration program [here](). (Note that, this process do not include camera calibration that is obtain the camera intrinsic parameters about lense distortion correction.)


## D. Run main.py for Dynamic Projection Mapping

You can choose some mode with flags like [here]().
