# Machine Learning Notes
These are assorted notes on topics from various courses and projects.

## 1. End-end-process
1. Data sourcing  - train_test_split
2. Exploratory data analysis - plot distributions, correlations
3. Data cleaning - drop redundant columns, handle missing data (drop or impute)
4. Feature engineering - one hot encoding categories, scaling/normalising numerical values, combining columns, extracting info from columns (e.g. zip code from address) 
5. Model selection and training - model and hyperparameters
6. Model tuning and evaluation metrics - classification: classification_report, confusion matrix, accuracy, recall, F1. Regression: error (RMSE, MAE, etc)
7. Predictions


## 2. Models

### 2.1. Machine learning categories

- Supervised learning
  - SVM
  - ...
- Unsupervised learning
  - Dimensionality reduction
  - Clustering
- Reinforcement Learning

For supervised learning:
- Regression
- Classification
  - Single class
  - Multi class


### 2.2. Machine learning models

#### 2.2.1 ANN (Artificial Neural Network)
General idea of a neural network that can then be extended to specific cases, e.g. CNNs and RNNs.

- Perceptron: a weighted average of inputs
- Network of perceptrons
- Activation function: a non-linear transfer function f(wx+b)
- Cost function and gradient descent: convex optimisation of a loss function using sub-gradient descent. The optimizer can be set in the compile method of the model.
- Backpropagation: use chain rule to determine partial gradients of each weight and bias. This means we only need a single forward pass followed by a single backward pass. Contrast this to if we perturbed each weight or bias to determine each partial gradient: in that case, for each epoch we would need to run a forward pass per weight/bias in the network, which is potentially millions! 
- Dropout: a technique to avoid overfitting by randomly dropping neurons in each epoch.

General structure:
1. Input layer
2. Hidden layer(s)
3. Output layer

Input and output layers are determined by the problem:
- Input size: number of features in the data
- Output size number of targets to predict, i.e. one for single class classification or single target regression, or multiple for multiclass (one per class)
- Output layer activation determined by problem. For single class classification `activation='sigmoid'`, for multiclass classification `activation='softmax'`
- Loss function determined by problem. For single class classification `loss='binary_crossentropy'`, for multiclass classification `loss='categorical_crossentropy'`

Hidden layers are less well-defined. Some heuristics here: https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw


Vanishing gradients can be an issue for lower layers in particular, where the partial gradients of individual layers can be very small, so when multiplied together in chain rule the gradient is vanishingly small.
Exploding gradients are a similar issue but where gradients get increasingly large when multiplied together.
Some techniques to rectify vanishing/exploding gradients:
- Use different activation functions with larger gradients close to 0 and 1, e.g. leaky ReLU or ELU (exponential linear unit)
- Batch normalisation: scale each gradient by the mean and standard deviation of the batch
- Different weight initialisation methods, e.g. Xavier initialisation


Example model outline
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()

# Input layer
model.add(Dense(78, activation='relu'))  # Number of input features
model.add(Dropout(0.2))  # Optional dropout layer after each layer. Omitted for next layers for clarity

# Hidden layers
model.add(Dense(39, activation='relu'))
model.add(Dense(19, activation='relu'))

# Output layer
model.add(Dense(1, activation='sigmoid'))  # Number of target classes (single class in this case)

# Problem setup
model.compile(loss='binary_crossentropy', optimizer='adam')  # Type of problem (single class classification in this case)

# Training
model.fit(
    x=X_train,
    y=y_train,
    batch_size=256,
    epochs=30,
    validation_data=(X_test, y_test)
)

# Prediction
model.predict(new_input)  # The new input needs to be shaped and scaled the sane as the training data
```


#### 2.2.2. CNN (Convolutional Neural Network)
These are used for image classification problems where convolutional filters are useful for extracting features from input arrays.

- Image kernels/filters
  - Grayscale 2D arrays
  - RGB 3D tensors
- Convolutional layers
- Pooling layers

General structure:
1. Input layer
2. Convolutional layer
3. Pooling layer
4. (Optionally more pairs of convolutional and pooling layers)
5. Flattening layer 
6. Dense hidden layer(s)
7. Output layer


Example model outline
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()

# Convolutional layer followed by pooling. Deeper networks may have multiple pairs of these.
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))  # Input shape determined by input data size
model.add(MaxPool2D(pool_size=(2, 2)))

# Flatten images from 2D to 1D before final layer
model.add(Flatten())

# Dense hidden layer
model.add(Dense(128, activation='relu'))

# Output layer
model.add(Dense(10, activation='softmax'))  # Size and activation determined by problem; multiclass classification in this case

# Problem setup
model.compile(loss='categorical_crossentropy', optimizer='adam')  # Loss determined by problem, multiclass classification in this case

# Training (with optional early stopping)
early_stop = EarlyStopping(monitor='val_loss',patience=2)
model.fit(x_train,y_cat_train,epochs=10,validation_data=(x_test,y_cat_test),callbacks=[early_stop])

# Prediction
model.predict(new_input)  # The new input needs to be shaped and scaled the sane as the training data
```

#### 2.2.3. RNN (Recurrent Neural Network)
These are used for modelling sequences with variable lengths of inputs and outputs.

Recurrent neurons take as their input the current data AND the previous epoch's output. 
We could try to pass EVERY previous epoch's output as an input, but would run into issues of vanushing gradients.

LSTMs take this idea a step further by incorporating the previous epochs input AND some longer lookback of epoch where some old outputs are "forgotten".

A basic RNN:
```

            --------------------------------
            |                              |
 H_t-1 ---> |                              |---> H_t
            |                              |
 X_t   ---> |                              |
            |                              |
            --------------------------------

where:
H_t is the neuron's output at current epoch t
H_t-1 is the neuron's output from the previous epoch t-1
X_t is the input data at current epoch t

H_t = tanh(W[H_t-1, X_t] + b)
```

An LSTM (Long Short Term Memory) unit:
```

            --------------------------------
            |                              |
 H_t-1 ---> |                              |---> H_t
            |                              |
 C_t-1 ---> |                              |---> C_t
            |                              |
 X_t   ---> |                              |
            |                              |
            --------------------------------
            
The `forget gate` determines which part of the old short-term memory and current input is "forgotten" by the new long-term memory.
These are a set of weights (between 0 and 1) that get applied to the old long-term memory to downweight it.
F_t = sigmoid(W_F[H_t-1, X_t] + b_F)

The `input gate` i_t similarly gates the input and old short-term memory.
This will later (in the update gate) get combined  with a candidate value for the new long-term memory `C_candidate_t`
I_t = sigmoid(W_I[H_t-1, X_t] + b_I)
C_candidate_t = tanh(W_C_cand[H_t-1, X_t] + b_C_cand) 

The `update gate` for the new long-term memory `C_t` is then calculated as a sum of forgotten old memory and input-weighted candidate memory:
C_t = F_t*C_t-1 + I_t*C_candidate_t 

The `output gate` O_t is a combination of the old short-term memory and latest input data.
This is then combined with the latest long-term memory to produce the output of the recurrent neuron, which is also the updated short-term memory H_t:
O_t = sigmoid(W_O[H_t-1, X_t] + b_O)
H_t = O_t * tanh(C_t) 

where:
H_t is short-term memory at epoch t
C_t is long-term memory at epoch t
X_t is the input data at current epoch t

sigmoid is a sigmoid  activation function
F_t is an intermediate forget gate weight
I_t is an intermediate input gate weight
O_t is an intermediate output gate weight
```


There are several variants of RNNs.
*RNNs with peepholes*
This leaks long-term memory into the forget, input and output gates.
Note that the forget gate and input gate each get the OLD long-term memory, whereas the output gate gets the NEW long-term memory.
```
F_t = sigmoid(W_F[C_t-1, H_t-1, X_t] + b_F)
I_t = sigmoid(W_I[C_t-1, H_t-1, X_t] + b_I)
O_t = sigmoid(W_O[C_t, H_t-1, X_t] + b_O)
```

*Gated Recurrent Unit (GRU)*
This combines the forget and input gates into a single gate. It also has some other changes.
This is simpler than a typical LSTM model as it has fewer parameters. This makes it more computationally efficient, and in practice they can have similar performance.
```
z_t = sigmoid(W_z[H_t-1, X_t])
r_t = sigmoid(W_r[H_t-1, X_t])
H_candidate_t = tanh(W_H_candidate[r_t*h_t-1, x_t])
H_t = (1 - z_t) * H_t-1 + z_t * H_candidate_t
```


The sequences modelled with RNNS can be:
- One-to-many
- Many-to-many
- Many-to-one


## A. Appendix
### A.1. Useful resources
Neural networks:
- Intuition behind neural networks http://neuralnetworksanddeeplearning.com/
- The deep learning bible (with lectures) https://www.deeplearningbook.org/
- Udemy course: https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp
- Heuristics for choosing hidden layers https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
- Alternatives to the deprecated model predict for different classification problems https://stackoverflow.com/questions/68776790/model-predict-classes-is-deprecated-what-to-use-instead

CNNs:
- Image kernels explained https://setosa.io/ev/image-kernels/
- Choosing CNN layers https://stats.stackexchange.com/questions/148139/rules-for-selecting-convolutional-neural-network-hyperparameters

RNNs:
- Overview of RNNs http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Wikipedia page contains the equations of LSTMs and peepholes https://en.wikipedia.org/wiki/Long_short-term_memory
- LSTMs vs GRUs https://datascience.stackexchange.com/questions/14581/when-to-use-gru-over-lstm 
- Worked example of LSTM http://blog.echen.me/2017/05/30/exploring-lstms/
