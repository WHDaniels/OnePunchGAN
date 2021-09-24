# OnePunchGAN
 
(Though the scope of the original project was too large to be completed in a reasonable amount of time, the result up to this point is a project in itself. Following in this repo's footsteps, I will be splitting up the originally intended endeavor into a series of projects to be released in the future.)

Much of the work in the area of image colorization through neural networks is based around gathering training data by converting colored images to greyscale. But in various other image domains, such as manga or line-art, the colored images cannot be converted to the original no-color image via simple greyscale conversion, as shading may be added to the image in the coloring process by the artist. In most cases, the original image is vastly different from a greyscale conversion of the artist's coloring. 

This project attempts to explore how we can still prevail in the creation of a colorization model given these restrictions in the context of a specific domain: manga\line art.

## Background

Greyscale to RGB colorization is a fairly well researched topic, and engaged with by those trying to learn more about computer vision through image generation by novel deep learning methods. But much of the associated research, tutorials, and projects are based on learning the relationship between a particular color combination (be it the A and B channels of the LAB color space or the RGB channels of the RGB color space) and the lightness channel of the original image. 

This is a exceedingly common practice due to the type of data most commonly used in colorization problems, which comes in the form of greyscale pictures. To train a colorization network, one need only obtain a RGB picture from a large dataset like ImageNet, and convert these images to a greyscale format. This greyscale format is essentially an encoding for the shading of the image. 

In practice this is a great way to gather training data, as pictures taken in greyscale (such as pictures from before color photography was widespread) and pictures converted to greyscale from color share the same domain space. In other words, a image taken by non-color photography and the same image but in color which is then converted to greyscale are the same image. The training data then consists of a converted-to-greyscale version as model input, and the original colored version as the target for the model to emulate.

[example images here]

## Problem

So now that we have this knowledge in mind, let's get colorings from our favorite image domains and convert them to greyscale to use as training data.

If we are pulling our images from an input and target domain that is fundamentally different from 

