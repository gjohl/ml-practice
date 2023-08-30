# Fast AI Part I
Notes from Fast AI Practical Deep Learning for Coders Part 1.

https://course.fast.ai/
https://github.com/fastai/fastbook/tree/master


## 1. Introduction to image classification models

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


## 2. Deployment
It can be useful to train a model on the data BEFORE you clean it
- Counterintuitive!
- The confusion matrix output of the learner gives you a good intuition about which classifications are hard
- `plot_top_losses` shows which examples were hardest for the model to classify. 
  This can find (1) when the model is correct but not confident, and (2) when the model was confident but incorrect
- `ImageClassifierCleaner` shows the examples in the training and validation set ordered by loss, so we can choose to keep, reclassify or remove them

For image resizing, random resize crop can often be more effective.
- Squishing can result in weird, unrealistic images
- Padding or mirroing can add false information that the model will erroneously learn
- Random crops give different sections of the image which acts as a form of data augmentation.
- `aug_transforms` can be use for more sophisticated data augmentation like warping and recoloring images.

A website for quizzes based on the book: www.aiquizzes.com

Hugging face spaces hosts models with a choice of pre-canned interfaces (Gradio in the example in the lecture)
to quickly deploy a model to the public. Streamlit is an alternative to Gradio that is more flexible.
https://huggingface.co/spaces

**Saving a model**
Once you are happy with the model you've trained, you can pickle the learner object and save it.
```
learn.export('model.pkl')
```

Then you can add the saved model to the hugging face space.
To use the model to make predictions
Any external functions you used to create the model will need to be instantiated too. 
```
learn = load_learner('model.pkl')
learn.predict(image)
```

Gradio requires a dict of classes as keys and probabilities (as floats not tensors) as the values.
To go from the Gradio prototype to a production app, you can view the Gradio API from the huggingface space which will show you the API.
The API exposes an endpoint which you can then hit from your own frontend app.

Github pages is a free and simple way to host a public website.
See this repo as an example of a minimal example html website which issues GET requests to the Gradio API https://github.com/fastai/tinypets

To convert a notebook to a python script, you can add `#|export` to the top of any cells to include in the script,
then use:
```
from nbdev.export import notebook2script
notebook2script('name_of_output_file.py')
```

Use `#|default_exp app` in the first cell of the notebook to set the default name of the exported python file.

How to choose the number of epochs to train for?
Whenever it is "good enough" for your use case.
If you need to train for longer, you may need to use data augmentation to prevent overfitting.
Keep an eye on the validation error to check overfitting.


## 3. Natural language processing