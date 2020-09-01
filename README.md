# Image Super-Resolution
The goal of this project is to research possible super-resolution methods, determine the most effective model for X-Ray medical imaging and evaluate this model 

SRCNN, ESPCNN, EDSR and MDSR (multiscale EDSR), RCAN and MRCAN (multiscale RCAN), and a Sharpening Module are implemented here

Models were trained on two datasets, `div2k` dataset for widespread image-resolution problem, and a group of several chest x-rays datasets for medical image super-resolution problem. There are folders, which contain parameters of trained models, the visualization of results and metrics evaluation for each problem. 

You can find the example of models application in **example.ipynb**

## Model description
SRCNN and ESPCNN represent the first generation of super-resolution models and have comparably low efficiency

EDSR and RCAN, as well as their multiscale analogues have much more complex structure and higher efficiency

Sharpening Module is a block, which can be applied for edge sharpening. It is not a network, usually it reduces metrics, but in some cases it could improve perceptual quality.

## Metrics
We evaluate the efficiency of models using 2 metrics: peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM)

## File description

**example.ipynb** demonstrates the usage of every model and contains some detail explanations

**images** contains images, used in **example.ipyng**

**chest** folder contains parameters for networks, trained on chest X-Ray datasets, results visualization with metrics evaluation, and train/validation splited filenames for data loading

**div2k** folder contains parameters for networks, trained on div2k dataset, results visualization with metrics evaluation and model comparison

**architectures.py** contains implementations of all the models available for usage to date

**dataLoader.py** contains PyTorch *DataLoader* class, which load and resize images from the dataset

**create_image.py** contains the function visualizing efficiency of a given net

**training.py** contains one-epoch train fuctions for single-scale and multi-scale training

**validation.py** contains metrics implementation and one-epoch validation for single-scale and multi-scale training

**readings.py** contains functions, used for data loading

**train.py** is a script, an example of training multi-scale RCAN model

**makeimage.py** is a script, an example of **create_image.py** usage for multi-scale RCAN model with scale equals 6

## Literature
- *Deep Learning for Single Image Super-Resolution: A Brief Review*. W. Yang, X. Zhang, Y. Tian, W. Wang, J. Xue, Q. Liao
- *Image super-resolution using deep convolutional networks*.
C. Dong, C. C. Loy, K. He, and X. Tang
- *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network*. W. Shi *et al*.
- *Enhanced Deep Residual Networks for Single Image Super-Resolution*. B. Lim, S. Son, H. Kim, S. Nah, K. M. Lee
- *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*. C. Ledig *et al*.
- *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks*. X. Wang *et al*.
- *Image Super-Resolution Using Very Deep Residual Channel Attention Networks*. Y. Zhang *et al*.