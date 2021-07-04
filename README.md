# Continuous Image-to-Image Translation Through Learning Fixed Points in GANs

This repository contains code for my degree project for the degree of MSc in Computer
Science. 

## Abstract
The field of image-to-image translation consists of learning a transformation of an image from one domain to another. It has experienced great success during recent years, with methods being able to generate realistic outputs when converting between multiple categorical domains at once. However, existing approaches have not yet been extended to ordinal domains. 
    
Therefore, this thesis investigates how existing image-to-image translation methods can be extended to use ordinal labels and introduces the Ordinal GAN (OrGAN) architecture as one possible solution to the problem. OrGAN is based on two fundamental modifications to existing methods, namely, adding Gaussian noise to the labels of the data samples in each iteration of training and using a pre-trained embedding network to feed embedded labels into the network instead of scalars. The effectiveness of the model is demonstrated empirically in a variety of synthetic data sets and compared to a direct application of an established work in categorical image-to-image translation. This shows that the presented methodology is a suitable starting point for future work within the field.

## Usage
1. Build the dockerfile
2. `cd` to the `src` directory and activate the `msc` environment in the docker container
3. Create a W&B sweep with e.g. `wandb sweep configs/experiments/clustered_hsv/organ/sweep.yaml`
4. Run the resulting sweep. Optionally change args in the argsfile in the sweep directory.
