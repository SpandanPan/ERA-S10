# Image Classification on CFAR10 using Pytorch

### Data Description
##### 1. Training DataSize : 50000 images
##### 2. Test DataSize: 10000 images
##### 3. Image Size: 3*32*32

### Data Augmentations Used
#### Library: albumentations
##### Transformations
##### 1. RadomCrop (with padding)
##### 2. Horizontal Flip
##### 3. Cutout
##### 4. Normalize


# Model Structure

##### PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
##### Layer1 
###### X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
###### R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
###### Add(X, R1)

##### Layer 2 
###### Conv 3x3 [256k]
###### MaxPooling2D
###### BN
###### ReLU

##### Layer 3 
###### X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
###### R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
###### Add(X, R2)
###### MaxPooling with Kernel Size 4
###### FC Layer 
###### SoftMax




#### Optimizers - Adam
#### One Cycle LP  - max LR 2.30E-02 (using LR finder)
![alt text](https://github.com/SpandanPan/ERA-S10/blob/main/LRFinder.png?raw=true)
#### Epoch 24

#### Accuracy
##### Training - 93.14
##### Test - 89.4
