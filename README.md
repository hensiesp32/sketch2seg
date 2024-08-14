# Sketch2Seg: Sketch-based Image Segmentation with Pre-trained Diffusion Model
This is the code repo for ICPR2024 paper `Sketch2Seg: Sketch-based Image Segmentation with Pre-trained Diffusion Model`

## Installation
To install the required python packages to run Sketch2Seg with conda
```
conda env create -f environment.yaml
conda activate control
```

## Evaluaton
To evaluation sketch2seg, please prepare validation dataset with `photo`,`sketch` and `seg annotation`,and replace in the `evaluation.py`.
To run a segmentation, you can run
```
python evaluation.py --exp_dir YOUR_CHECKPOINT_PATH --save_dir YOUR_SAVE_PATH
```
## Train
If you want to train your own model, please prepare your training data as well, then you can run
```
python train_classifier.py --exp_dir PATH_TO_SAVE_CHECKPOINT 
```

## SkecthyCOCOSeg dataset
You can get the dataset at [here](https://drive.google.com/file/d/17KTH37dxQrVl1APAkobC8L-B-c5gJ0jH/view?usp=drive_link)
We provide the 256*256 resolution version,if you need other resolutions,please contact us.