# Deep-learning-for-in-vivo-near-infrared-imaging

This reposotory is our project for the paper "Deep learning for in vivo near-infrared imaging".  
The code is based on the original implementation of CycleGAN and pix2pix https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. We simplified the code for our task.  

We trained artificial neural networks to transform a fluorescence image in the shorter wavelength NIR window of 900-1300 nm (NIR-I/IIa) to an image resembling a NIR-IIb (1500-1700 nm) image. Details of the experiments can be found in our paper.  

## Model Architecture
CycleGAN:  
![image of CycleGAN](https://github.com/zhuoranzma/Deep-learning-for-in-vivo-near-infrared-imaging/blob/master/figs/CycleGAN.png) 
pix2pix:  
![image of pix2pix](https://github.com/zhuoranzma/Deep-learning-for-in-vivo-near-infrared-imaging/blob/master/figs/pix2pix.png)  


## Dataset
Structure of the dataset:  
```
datasets/  
    NIRI_to_NIRII/  
        train/  
            A  
            B  
        val/  
            A  
            B  
```
A and B contain NIR-I and NIR-IIb images, respectively. To train the model, put images in the train directory. To test the model, put the images in the val directory.  

## Train the model
To train the CycleGAN model, run:  
```
python train.py --u_net --cuda
```
parameters will be saved in the model directory

## Test the model
To evaluate the model on the evaluation set, run:
```
python test.py --u_net --cuda
```
We provide pre-trained weights of the U-Net generator and PatchGAN discriminators in the model directory.
