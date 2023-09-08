# Fast AI Part I
Notes from Fast AI Practical Deep Learning for Coders Part 1.

https://course.fast.ai/
https://github.com/fastai/fastbook/tree/master


## 1. Introduction to image classification models

> Homework task:
> 
> Train an image classifier
> https://github.com/gjohl/ml-practice/blob/master/ml-practice/notebooks/fastai/1_image_classifier.ipynb

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
> Homework task:
> 
> Deploy a model to Huggingface Spaces
> https://huggingface.co/spaces/GurpreetJohl/binary_image_classifier_vw_rr
> 
> Deploy a model to a Github Pages website
> https://github.com/gjohl/vw_classifier

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


## 3. How does a neural net learn?
> Homework task:
> Recreate the spreadsheet to train a linear model and a neural network from scratch
> https://docs.google.com/spreadsheets/d/1hma4bTEFuiS483djqE5dPoLlbsSQOTioqMzsesZGUGI/edit?usp=sharing

Options for cloud environments: Kaggle, Colab, Paperspace 

Comparison of performance vs training time for different image models: https://www.kaggle.com/code/jhoward/which-image-models-are-best/
Resnet and convnext are generally good to start with.
The best practice is to start with a "good enough" model with a quick training time, so you can iterate quickly.
Then as you get further on with your research and need a better model, you can move to a slower, more accurate model.
The criteria we generally care about are:
1. How fast are they
2. How much memory do they use
3. How accurate are they

A learner object contains the pre-processing steps and the model itself.

**Fit a quadratic function** 
How do we fit a function to data? An ML model is just a function fitted to data. Deep learning is just fitting an infinitely flexible model to data.
In [this notebook](https://www.kaggle.com/code/gurpreetjohl/how-does-a-neural-net-really-work) there is an interactive cell to 
fit a quadratic function to some noisy data: `y = ax^2 +bx + c`
We can vary a, b and c to get a better fit by eye.
We can make this more rigorous by defining a loss function to quantify how good the fit is.
In this case, use the mean absolute error `mae = mean(abs(actual - preds))`
We can then use stochastic gradient descent to autome the tweaking of a, b and c to minimise the loss.

We can store the parameters as a tensor, then pytorch will calculate gradients of the loss function based on that tensor.
```
abc = abc = torch.tensor([1.1,1.1,1.1]) 
abc.requires_grad_()  # Modifies abc in place so that it will include gradient calculations on anything which uses abc downstream
loss = quad_mae(abc)  # loss uses abc so we will get the gradient of loss too
loss.backward()  # Back-propagate the gradients
abc.grad  # Returns a tensor of the loss gradients
```

We can then take a "small step" in the direction that will decrease the gradient. 
The size of the step should be proportional to the size of the gradient.
We define a learning rate hyperparameter to determine how much to scale the gradients by.
```
learning_rate = 0.01
abc -= abc.grad * learning_rate
loss = quad_mae(abc)  # The loss should now be smaller
```

We can repeat this process to take multiple steps to minimise the gradient.
```
for step in range(10):
    loss = quad_mae(abc)
    loss.backward()
    with torch.no_grad():
        abc -= abc.grad * learning_rate
    print(f'step={step}; loss={loss:.2f}')
```

The learning rate should decrease as we get closer to the minimum to ensure we don't overshoot the minimum and increase the loss.
A learning rate schedule can be specified to do this.


**Fit a deep learning model**
For deep learning, the premise is the same but instead of quadratic functions, we fit ReLUs and other non-linear functions.
[Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) states that this is infinitely
expressive if enough ReLUs (or other non-linear units) are combined.
This means we can learn any computable function.

A ReLU is essentially a linear function with the negative values clipped to 0
```
def relu(m,c,x):
    y = m * x + c
    return np.clip(y, 0.)
```

This is all that deep learning is! All we need is:
1. A model - a bunch of ReLUs combined will be flexible
2. A loss function - mean absolute error between the actual data values and the values predicted by the model
3. An optimiser - stochastic gradient descent can start from random weights to incrementally improve the loss until we get a "good enough" fit

We just need enough time and data.
There are a few hacks to decrease the time and data required:
- Data augmentation
- Running on GPUs to parallelise matrix multiplications
- Convolutions to skip over values to reduce the number of matrix multiplications required
- Transfer learning - initialise with parameters from another pre-trained model instead of random weights.

[This spreadsheet](https://docs.google.com/spreadsheets/d/1hma4bTEFuiS483djqE5dPoLlbsSQOTioqMzsesZGUGI/edit?usp=sharing)
is a worked example of manually training a multivariate linear model, then extending that to a neural network summing two ReLUs. 


## 4. Natural language processing
> Homework:
> 
> Kaggle NLP pattern similarity notebook https://www.kaggle.com/code/gurpreetjohl/getting-started-with-nlp-for-absolute-beginners/edit

NLP applications: categorising documents, translation, text generation.

Using [Huggingface transformers](https://huggingface.co/docs/transformers/index) library for this lesson.
It is now incorporated into the fastai library.

ULMFit is an algorithm which uses fine-tuning, in this example to train a positve/negative sentiment classifier in 3 steps:
1. Train an RNN on wikipedia to predict the next word. No labels required.
2. Fine-tune this for IMDb reviews to predict the next word of a movie review. Still no labels required.
3. Fine-tune this to classify the sentiment.

Transformers have overtaken ULMFit as the state-of-the-art.

Looking "inside" a CNN, the first layer contains elementary detectors like edge detectors, blob detectors, gradient detectors etc.
These get combined in non-linear ways to make increasingly complex detectors. Layer 2 might combine vertical and horizontal edge detectors into a corner detector.
By the later layers, it is detecting rich features like lizard eyes, dog faces etc. 
See: https://arxiv.org/abs/1311.2901

For the fine-tuning process, the earlier layers are unlikely to need changing because they are more general. So we only need to fine-tune (AKA re-train)
the later layers.


**Kaggle competition walkthrough**

https://www.kaggle.com/code/gurpreetjohl/getting-started-with-nlp-for-absolute-beginners/edit

Reshape the input to fit a standard NLP task
- We want to learn the similarity between two fields and are provided with similarity scores.
- We concat the fields of interest (with identifiers in between). The identifiers themselves are not important, they just need to be consistent.
- The NLP model is then a supervised regression task to predict the score given the concatendated string.

`df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor`

**Tokenization:**
Split the text into tokens (words).
Tokens are, broadly speaking, words.
There are some caveats to that, as some languages like Chinese don't fit nicely into that model.
We don't want the vocabulary to be too big.
In practice, we tokenize into subwords.

**Numericalization:**
Map each unique token to a number. One-hot encoding.

The choice of tokenization and numericalization depends on the model you use.
Whoever trained the model chose a convention for tokenizing.
We need to be consistent with that if we want the model to work correctly.

**Models:**
The Huggingface model hub contains thousands of pretrained models https://huggingface.co/models
For NLP tasks, it is useful to choose a model that was trained on a similar corpus, so you can search the model hub.
In this case, we search for "patent".

Some models are general purpose, e.g. deberta-v3 used in the lesson.


ULMFit handles large documents better as it can split up the document.
Transformer approaches require loading the whole document into GPU memory, so struggle for larger documents.

**Overfitting:**
If a model is too simple (i.e. not flexible enough) then it cannot fit the data and be biased. Underfitting.

If the model fits the data points too closely, it is overfitting.

A good validation set, and monitoring validation error rather than training error as a metric, is key to avoiding overfitting.
https://www.fast.ai/posts/2017-11-13-validation-sets.html

Often people will default to using a random train/test split (this is what scikit-learn uses).
This is a BAD idea very often.
For time-series data, it's easier to infer gaps than it is to predict a block in the future. The latter is the more common task but a random split simulates the former, giving unrealistically good performance.
For image data, there may be people, boats, etc that are in the training set but not the test set. By failing to have new people in the validation set, the model can learn things about specific people/boats that it can't rely on in practice.

**Metrics vs loss functions:**
Metrics are things that are human-understandable.
Loss functions should be smooth and differentiable to aid in training.

These can sometimes be the same thing, but not in general.
For example, accuracy is a good metric in image classification.
We could tweak the weights in such a way that it improves the model slightly, but not so much that it now correctly classifies a previously incorrect image.
This means the metric function is bumpy, therefore a bad loss function.
https://www.fast.ai/posts/2019-09-24-metrics.html

AI can be particularly dangerous at confirming systematic biases, because it is so good at optimising metrics, so it will
conform to any biases present in the training data. MAking decisions based on the model then reinforces those biases.
- Goodhart's law applies: If a metric becomes a target it's no longer a good metric

**Correlations**
The best way to understand a metric is not to look at the mathematical formular, but to plot some data for which the metric
is high, medium and low, then see what that tells you.

After that, look at the equation to see if your intuition matches the logic.

**Choosing a learning rate**
Fast AI has a function to find a good starting point.
Otherwise, pick a small value and keep doubling it until it falls apart.


## 5. From scratch model
Train a linear model from scratch in Python using on the Titanic dataset.
Then train a neural network from scratch in a similar way.
This is an extension of the spreadsheet approach to make a neural network from scratch in lesson 3.

**Imputing missing values**
Never throw away data.
An easy way of imputing missing values is to fill them with the mode.
This is good enough for a first pass at creating a  model.

**Scaling values**
Numeric values that can grow exponentially like prices or population sizes often have long-tailed distributions.
An easy way to scale is to take log(x+1). The `+1` is just to avoid taking log of 0.

**Categorical variables**
One-hot encode any categorical variables.
We should include an "other" category for each in case the validation or test set contains a category we didn't encounter in the training set.
If there are categories with ony a small number of observations we can group them into an "other" category too.

**Broadcasting**
Broadcasting arrays together avoids boilerplate code to make dimensions match.
https://numpy.org/doc/stable/user/basics.broadcasting.html
https://tryapl.org/

**Sigmoid final layer**
For a binary classification model, the outputs should be between 0 and 1.
If you train a linear model, it might result in negative values or values >1.
This means we can improve the loss by just clipping to ensure they stay between 0 and 1.
This is the idea behind the sigmoid function for output layers: smaller values will tend to 0 and larger values will tend to 1.
In general, for any binary classification model, we should always have a sigmoid as the final layer.
If the model isn't training well, it's worth checking that the final activation function is a sigmoid.

Function: `y = 1 / (1+e^-x))`

**Focus on input and output layers**
In general, the middle layers of a neural network are similar between different problems.
The input layer will depend on the data for our specific problem.
The output will depend on the target for our specific problem.
So we spend most of our time thinking about the correct input and output layers, and the hidden layers are less important.

**Using a framework**
When creating the models from scratch, there was a lot of boilerplate code to:
- Impute missing values using the "obvious" method (fill with mode)
- Normalise continuous variables to be between 0 and 1
- One hot encode categorical variables
- Repeat all of these steps in the same order for the test set

The benefits of using a framework like fastai:
- Less boilerplate, so the obvious things are done automatically unless you say otherwise
- Repeating all of the feature engineering steps on the output is trivial with DataLoaders

**Ensemble**
Creating independent models and taking the mean of their predictions improves the accuracy.
There are a few different approaches to ensembling a categorical variable:
1. Take the mean of the predictions (binary 0 or 1)
2. Take the mean of the probabilities (continuous between 0 and 1) then threshold the result
3. Take the mode of the predictions (binary 0 or 1)

In general the mean approaches work better but there's no rule as to why, so try all of them.


## 6. Random forests
It's worth starting with a random forest as a baseline model because "it's very hard to screw up".

A random forest is an ensemble of trees.
A tree is an ensemble of binary splits.
`binary split -> tree -> forest`

**Binary split**
Pick a column of the data set and split the rows into two groups.
Predict the outcome based just on that.
For example, in the titanic dataset, pick the Sex column. This splits into male vs female. If we predict that all female passengers survived and all male passengers died, that's a reasonably accurate prediction.