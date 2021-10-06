# OnePunchGAN
 
(Though the scope of the original project was too large to be completed in a reasonable amount of time, the result up to this point is a project in itself. I'll be updating this repo with additions that move towards the goal of the long-term project, which was the formulation of an acceptable colorization model for a specific manga\line-art series.)

Much of the work in the area of image colorization through neural networks is based around gathering training data by converting colored images to greyscale. But in various other image domains, such as manga or line-art, the colored images cannot be converted to the original no-color image via simple greyscale conversion, as shading (the L channel in the case of a LAB color space) may be added to the image in the coloring process by the artist. In most cases, the original no-color image is different from a greyscale conversion of the artist's coloring.

This project attempts to explore how we can still prevail in the creation of a colorization model given these restrictions in the context of a specific domain: manga\line art.

## Background

Automactic image colorization is a fairly well researched topic, and is one engaged with by those trying to learn more about computer vision through image generation by novel deep learning methods. But much of the associated research, tutorials, and projects are based on learning the relationship between a particular color combination (usually the A and B channels from the LAB color space) and the lightness channel of the original image. 

This is a exceedingly common practice due to the type of data most commonly used in colorization problems, which comes in the form of greyscale pictures. To train a colorization network, one need only obtain a RGB picture from a large dataset like ImageNet, and convert these images to a greyscale format. This greyscale format is essentially an encoding for the shading of the image.

[color image here], [greyscale image here]

In practice this is a great way to gather training data, as pictures taken in greyscale (such as pictures from before color photography was widespread) and pictures converted to greyscale from color share the same domain space. In other words, a image taken by non-color photography and the same image but in color which is then converted to greyscale are the same image. The training data then consists of a converted-to-greyscale version as model input, and the original colored version as the target for the model to emulate.

[example images here]

## Problem

(Explain the problem with suing this technique with this domain space)
[Pictures to explain]

## Method

(What was the method used to overcome the problem mentioned above?)
[Architecture, example pictures, etc]

# Training

(Hyperparameters, training, method, data gathered)
[images representing this]

# Results (so far)

[Images so far, sorted by image size and epoch, etc]

# Conclusions (any more to add?)

(end with conclusion if there is not anything left of vlaue to add)
