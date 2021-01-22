# Continuous Image-to-Image Translation Through Learning Fixed Points in GANs

This repository contains code for my degree project for the degree of MSc in Computer
Science. 

## Research Question
The research project is based on investigating:

- Can continuous image-to-image translation be performed effectively through use of 
  learning fixed points as in Siddiquee et al. (2019)?
- Is the feature space that is learned through learning fixed points as in Siddiquee et 
  al. (2019) smoothly interpolateable between real continuous labels when trained on 
  *real continuous* labels?
- If the learned feature space is smooth, can the method be applied to continuous-valued
  image deblurring?
- Is the feature space that is learned through learning fixed points as in Siddiquee et 
  al. (2019) smoothly interpolateable between real continuous labels when trained on 
  *discrete* labels?

## Background
Siddiquee et al. (2019) provide an architecture entitled Fixed-Point GAN that is shown 
to be an effective means of performing class-conditional image-to-image translations 
while changing a minimal subset of the image, leaving the non-class specific segments of
the image as fixed points in the translation. The core component of the architecture is
a set of loss functions and a training procedure that facilitates the learning of this
task, while the specific layer architecture is left as unimportant.

Siddiquee et al. (2019) evaluate their architecture on images of human faces from the
CelebA dataset where the labels are multi-hot encoded binary attributes such as gender
or presence of eyeglasses. Siddiquee et al. (2019) also present a medical application of
the architecture by identifying and localizing brain lesions in images with binary
labels indicating the presence of lesions in the image.

While the results shown by Siddiquee et al. (2019) are impressive and of particular
interest to those who work in the field of machine learning for medicine, they are
limited by only being tested on categorical labels. Extending their work to discrete or
continuous labels provides a slew of clear benefits, as the architecture could be
applied to any number of tasks such as selective blurring or deblurring (e.g. simulating
different aperture sizes and the effect this has on depth-of-field), modifying the
relative size of and distance between facial features, or applying the architecture on
medical tasks such as kidney health which are graded on multiple-step discrete scales. 

There has been some other work that has been done on continuous conditional GANs.
Marriott et al. (2018) attempt to enable more fine-grained conditional generation of
images by extending the binary labels of the training data with additional attributes
which are 0 if the actual label is 0 and random continuous values otherwise. By
adjusting these free attributes different variations of how the original label can be
met can be generated, while an additional identity constraint ensures that variations in
the free attributes preserve the overall structure of the image. Some example
applications by Marriott et al. (2018) include changing the lighting of images or the
pose of the subject of an image. Marriott et al. (2018) show that the impact of each
additional parameter is disentangled, and that continuous values for the parameters can
lead to continuous changes in the output. However, the specific disentangled effect of
each parameter is learned in an unsupervised fashion, meaning that the latent space has
to be explored manually to determine an interpretation of what each parameter changes. 

One recent approach that explicitly models continuous labels is provided by Ding  et 
al.  (2020). Their work strives to overcome the issues of not having data available for
each possible continuous label value as well as supporting a potentially infinite number
of possible values in the continuous regression's target domain by incorporating random
noise as an augmentation to each label and introducing new loss functions.

Both the work by Marriott et al. (2018) and the work by Ding  et  al.  (2020) are purely
conditioned on the class label that should be generated. One of the appealing properties
of the Fixed-Point GAN architecture (Siddiquee  et  al.  2019) is that it is jointly
conditioned on an original image and a class in order to perform image-to-image
translations, something that the other two methods are not designed to do. As such, it
would be interesting to evaluate the performance of the Fixed-Point GAN architecture
(Siddiquee  et  al.  2019) on continuous image-to-image translation tasks and
potentially extend the method to better handle the new task.

An effective continuous image-to-image translation method can be of interest to a broad
variety of use cases. The research group that I have discussed the project with at KTH
are largely interested in the medical applications of the project, such as the
aforementioned case of grading kidney health. 

If a meaningful continuous latent space can be formed around the discrete labels in a
dataset, the method could provide a powerful data augmentation tool where minor
adjustments to the class labels could be made to generate a series of new images before
feeding them into training a different network. This could be meaningful for a variety
of computer vision tasks.

## Bibliography
- Ding, X., Wang, Y., Xu, Z., Welch, W. J. & Wang, Z. J. (2020), "Ccgan: Continuous 
  conditional generative adversarial networks for image generation", arXiv preprint 
  arXiv:2011.07466.
- Marriott, R. T., Romdhani, S. & Chen, L. (2018), "Intra-class variation isolation in 
  conditional gans",arXiv preprint arXiv:1811.11296.
- Siddiquee, M. M. R., Zhou, Z., Tajbakhsh, N., Feng, R., Gotway, M. B., Bengio,  Y.  
  &  Liang,  J.  (2019),  Learning  fixed  points  in  generative  adversarial networks:
  From  image-to-image  translation  to  disease  detection  and  localization, in 
  "Proceedings of the IEEE International Conference on Computer Vision", pp. 191–200.
