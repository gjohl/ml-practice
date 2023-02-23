# Machine Learning Notes
These are assorted notes on topics from various courses and projects.

## 1. End-end-process
1. Data sourcing  - train_test_spllit
2. Exploratory data analysis - plot distributions, correlations
3. Data cleaning - drop redundant columns, handle missing data (drop or impute)
4. Feature engineering - one hot encoding categories, scaling/normalising numerical values, combining columns, extracting info from columns (e.g. zip code from address) 
5. Model selection and training - model and hyperparameters
6. Model tuning and evaluation metrics - classification: classification_report, confusion matrix, accuracy, recall, F1. Regression: error (RMSE, MAE, etc)
7. Predictions


### 1.1 Data sourcing
[//]: # (TODO section on each part of the list above)

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

- Perceptron
- Network of perceptrons
- Activation function
- Cost function and gradient descent
- Backpropagation
- Dropout

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
5. Dense hidden layer
6. Output layer


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
model.compile(loss='categorical_crossentropy', optimizer='adam')  # Loss determined by problem, multiclass slassification in this case

# Training (with optional early stopping)
early_stop = EarlyStopping(monitor='val_loss',patience=2)
model.fit(x_train,y_cat_train,epochs=10,validation_data=(x_test,y_cat_test),callbacks=[early_stop])

# Prediction
model.predict(new_input)  # The new input needs to be shaped and scaled the sane as the training data
```

## A. Appendix
### A.1. Useful resources
Neural networks:
- Intuition behind neural networks http://neuralnetworksanddeeplearning.com/
- The deep learning bible (with lectures) https://www.deeplearningbook.org/
- Udemy course: https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp
- Heuristics for choosing hidden layers https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

CNNs:
- Image kernels explained: https://setosa.io/ev/image-kernels/
