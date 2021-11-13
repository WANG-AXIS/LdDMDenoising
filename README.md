# Impact of loss functions on the performance of a deep neural network designed to restore low-dose digital mammography

======

This repository contains the training and testing codes for the paper "[Impact of loss functions on the performance of a deep neural network designed to restore low-dose digital mammography]()". For simulating dose reduction on clinical images, we used the codes available [here](https://lucasbusp.wixsite.com/lucasborges/simulation-of-dose-reduction). Also, we used a model-based (MB) restoration as a benchmark, also available [here](https://lucasbusp.wixsite.com/lucasborges/image-restoration), which uses the commonly known [BM3D](https://webpages.tuni.fi/foi/GCF-BM3D/). 

## Network architecture:

![](imgs/breast_dm_network.png)

## Some results:

Restoration of images with a dose reduction factor of 75%:
![](imgs/75.png)

Restoration of images with a dose reduction factor of 50%:
![](imgs/50.png)

## Reference:

If you use the toolbox, we will be very grateful if you refer to this [paper]():


---
AI-based X-ray Imaging System ([AXIS](https://wang-axis.github.io))  
Department of Biomedical Engineering  
Rensselaer Polytechnic Institute  
Troy - USA  

Laboratory of Computer Vision ([Lavi](http://iris.sel.eesc.usp.br/lavi/))  
Department of Electrical and Computer Engineering  
São Carlos School of Engineering, University of São Paulo  
São Carlos - Brazil
