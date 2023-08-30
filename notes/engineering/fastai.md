# Fast AI Part I
Notes from Fast AI Practical Deep Learning for Coders Part 1.

https://course.fast.ai/
https://github.com/fastai/fastbook/tree/master


## 1. Intro

Ethics course https://ethics.fast.ai/

Research on education:
Coloured cups - green, amber, red - 
Meta learning by Radek Osmulski
Mathematician's Lament by Paul Lockhart
Making Learning Whole by David Perkins

Before deep learning, the approach to machine learning was to enlist many domain experts to handcraft features
and feed this into a constrained linear model (e.g. ridge regression).
This is time-consuming, expensive and requires many domain experts.
Neural networks learn these features. 
Looking inside a CNN, for example, shows that these learned features match interpretable features that an expert might handcraft.

For image classifiers, you don't need particularly large images as inputs.
GPUs are so quick now that if you use large images, most of the time is spent on opening the file rather than computations.
So often we resize images down to 400x400 pixels or smaller.

For most use cases, there are pre-trained models and sensible default values that we can use.
In practice, most of the time is spent on the input layer and output layer. For most models the middle layers are identical.

Data blocks structure the input to learners.
`DataBlock` class:
- `blocks` determines the input and output type as a tuple. For multi-target classification this tuple can be arbitrary length.
- `get_items` - function that returns a list of all the inputs
- `splitter` - how to split the training/validation set
- `get_y` - function that returns the label of a given input image
- `item_tfms` - what transforms to apply to the inputs before training, e.g. resize
- `dataloaders` - method that parallelises loading the data.

A learner combines the model (e.g. resnet or something from timm library) and the data to run that model on (the dataloaders from the DataBlock).
- `fine_tune` method starts with a pretrained model weights rather than randomised weights, and only needs to learn the differences between your data and the original model.

Other image problems that can utilise deep learning
- Image classification
- Image segmentation

Other problem types use the same process, just with different DataBlock `blocks` types and the rest is the same.
For example, tabular data, collaborative filtering.

RISE is a jupyter notebook extensions to turn notebooks into slides.
Jeremy uses notebooks for: source code, book, blogging, CI/CD.

Traditional computer programs are essentially:
```
inputs ---> program ---> results
```

Deep learning models are:
```
inputs ---> model ---> results ---> loss
            ^                         |
weights ----|                         |
 ^                                    |
 |--------------(update)--------------|
```


## 2.
