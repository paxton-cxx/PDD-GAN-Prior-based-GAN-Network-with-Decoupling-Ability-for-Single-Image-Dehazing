# PDD-GAN-Prior-based-GAN-Network-with-Decoupling-Ability-for-Single-Image-Dehazing
please ensure install the packages- ```pytorch```,```pytorch-lightning```,```opencv```,```numpy```.
 The conda packages are provided in the ```requirements.txt```.

## The followings are some points we need to declare:
##### 1. Our model is in ```./model/PDD.py```.
##### 2. ```main.py``` is used to train our model.
##### 3. ```test.py``` is used to evaluate the metrics of SOTS.
##### 4. ```test2.py``` is used to generate dehazed images.
##### 5. ```./dataset/data_final.py``` is used to provided training data, you need to write your own like ours.
##### 6. ```./dataset/data_test_sots_1.py``` is used to test. Here the width and height of the image should be a multiple of 16.
This file uses the resize function to make it. And ```./dataset/data_test_sots_2.py``` uses the pad function.We recommend the second method.
##### 7. All the dataset files you need to write yourself or follow as what we did.
