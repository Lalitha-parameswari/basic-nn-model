# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
The Neural network model contains input layer,two hidden layers and output layer.Input layer contains a single neuron.Output layer also contains single neuron.First hidden layer contains nine neurons and second hidden layer contains seven neurons.A neuron in input layer is connected with every neurons in a first hidden layer.Similarly,each neurons in first hidden layer is connected with all neurons in second hidden layer.All neurons in second hidden layer is connected with output layered neuron.Relu activation function is used here .It is linear neural network model. Data is the key for the working of neural network and we need to process it before feeding to the neural network. In the first step, we will visualize data which will help us to gain insight into the data.We need a neural network model. This means we need to specify the number of hidden layers in the neural network and their size, the input and output size.Now we need to define the loss function according to our task. We also need to specify the optimizer to use with learning rate.Fitting is the training step of the neural network. Here we need to define the number of epochs for which we need to train the neural network.After fitting model, we can test it on test data to check whether the case of overfitting.

## Neural Network Model
![image](https://user-images.githubusercontent.com/103946827/187207180-b8a7089e-b985-4b04-8eb8-bc5b310c5bfe.png)



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
# Developed by: C.Lalitha Parameswari
# Registration number: 212219220027

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data=pd.read_csv("Book1new.csv")
data.head()
x=data[['input']].values
x
y=data[['output']].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=40)

scaler=MinMaxScaler()
scaler.fit(x_train)
scaler.fit(x_test)
x_train1=scaler.transform(x_train)
x_test1=scaler.transform(x_test)

AI=Sequential([
    Dense(9,activation='relu'),
    Dense(7,activation='relu'),
    Dense(1)
])
AI.compile(optimizer='rmsprop',loss='mse')
AI.fit(xtrain1,ytrain,epochs=2000)
loss_df=pd.DataFrame(AI.history.history)
loss_df.plot()
AI.evaluate(x_test1,y_test)

x_n1=[[29]]
x_n1_1=Scaler.transform(x_n1)
AI.predict(x_n1_1)
```


## Dataset Information
![image](https://user-images.githubusercontent.com/103946827/187208971-5941300d-aea2-4fe7-be87-cdfbad916152.png)


## OUTPUT


### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/103946827/187209170-6612903d-21a0-44a8-825d-0c39a01e9006.png)


### Test Data Root Mean Squared Error
![image](https://user-images.githubusercontent.com/103946827/187209403-bb15ab7d-bd5d-4b77-87ec-d912530645fe.png)


### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/103946827/187209649-81ae125f-5e8f-4bda-aa78-0d44fb5fe842.png)


## RESULT
Thus,the neural network regression model for the given dataset is developed.

