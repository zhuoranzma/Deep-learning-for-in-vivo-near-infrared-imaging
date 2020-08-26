# Deep-learning-for-in-vivo-near-infrared-imaging

This reposotory is our project for the paper "Deep learning for in vivo near-infrared imaging".  
The code is based on the original implementation of CycleGAN and pix2pix https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. We simplified the code for our task.  

We trained artificial neural networks to transform a fluorescence image in the shorter wavelength NIR window of 900-1300 nm (NIR-I/IIa) to an image resembling a NIR-IIb (1500-1700 nm) image. Details of the experiments can be found in our paper.  

## Model Architecture
![image of CycleGAN](https://github.com/zhuoranzma/Deep-learning-for-in-vivo-near-infrared-imaging/blob/master/figs/CycleGAN.png)  
![image of pix2pix](https://github.com/zhuoranzma/Deep-learning-for-in-vivo-near-infrared-imaging/blob/master/figs/pix2pix.png)
