Pytorch
Summary
Py torch uses tensors
Tensors are like numpy arrays
Tensors can be used to determin the gradient or the first differential of a function to find the minimum - ex gradient descent


ANN
define the model
(1) initialize and fix the shape of input hidden and output layers
(2) define a forward function to set the activation function and relationships between the layers

Define loss function (MSE etc)
Define criterion function used for optimizing such as Gradient descent (stochastic gradient descent), Adam etc. Pass the model parameters to the gradient descent for locking on to the weight of the network

for eah epoch
model.forward()
optimizer.zero_grad() # creates the gradient function by differentiating it wrt forward
loss.backward() # this propogates the weights backwards to get the weights
optimizer.step() # this updates the parameters


CNN
Model - a) while initiating include multiple convolutional layers - these are filters for feature extraction the weight that the system learns b) create pooling layers during forward pass to reduce the dimensions c) typical network will have set of conv layers and also linear layers d) softmax is used in last layer for classification

RNN
In LSTM the last few records are first split into the validation split
The training is then split into batches of X and y starting from the earliest time series record
The output from lstm , the last value is taken
LSTM works best wehn the inputs and outputs are scaled (MinMaxScaler -1,1)
Typical loss is MSELoss()
for each epoch: (Many times over all the data)
for each tuple of input and output - say 12 month bundles if seasonal: Train is usually a set of seq and label tuples
From the output take only last prediction


Each LSTM layer has 3 inputs and 3 outputs
Input t
Short term mem input C(t-1)
Long Term mem input h(t-1)


Output t
Short term mem output C(t)
Long Term mem output h(t)

LSTM can be used for time series and also for text prediction





---------------------
# For ANN use the 02-ANN-Artificial-Neural-Networks  folder
    within that he data models already defined - use the following as templates
    04a-Full-ANN-Code-Along-Regression.ipynb - for regression
    04b-Full-ANN-Code-Along-Classification.ipynb - for classification




# Installation
conda env create -f pytorch_course_env.yml

# Activating the environment and later deactivating it
conda activate pytorch_env
conda deactivate

- The name of the environment , packages and key dependencies are installed in the yml file


# Torch basics
Tensor is a multi dimensional array (Scalars, Vectors, Martix etc are 0-D , 1-D and 2-D tensors)


import torch

#### with a link to the numpy arrayi.e. changes if the array changes
torch.from_numpy(array_var)
torch.as_tensor(array_var)

#### with-OUT a link to the numpy arrayi.e. does not change if the array changes
torch.tensor(array_var)  [ Creates the same as base type i.e. retains as int32 if integers]
torch.Tensor(array_var) (or) torch.FloatTensor(array_var) [ Creates the same with a change to float]

torch.empty(4,2) // Creates a empty float memory hold

## torch allows all the numpy functions - very similar except here torch.
torch.zeros(4,2) // floating point zeros
torch.zeros(4,2, dtype=torch.int32) // int32 etc
torch.ones(2,2)
torch.arange(0,20,2).reshape(5,2)
torch.linspace(0.18.12).reshape(3,4)
torch.rand(4,3)
torch.randn(4,3) // std normal dist i.e. mean 0 and sigma =1

Change the data type using the following function
my_tensor = my_tensor.type(torch.int32)

Create random numbers from the shape of another tensor say x
torch.rand_like(x) or torch.randn_like(x) or torch.randint_like(x, low = 0, high=11)

For plots -convert tensors to numpy using


Document operations in tensors (Lost Unsaved)


Gradient
x = torch.tensor(2., requires_grad=True)
y = x**2 + 4*x**3 + 5*x + 1
y.backward()
x.grad  // This is gradient with respect to x and put the x value



ANN
import torch.nn as nn

# Generate data
x = torch.arange(1,51,1).reshape(-1,1).type(torch.float)
e = torch.randn(50).reshape(-1,1)
y = 2.*x + 1. + e

# define model layer
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model(1,1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 1e-3)

# Run each through and epoch and see the loss
y_predicted = model.forward(x)
loss = criterion(y_predicted,y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
print(loss.item())

model.linear.weight.item()
model.linear.bias.item()

plt.plot(x.numpy(),y.numpy())
plt.plot(x.numpy(),y_predicted.detach().numpy(),'r')


# Data sets and train test splits
from torch.utils.data import TensorDataset , DataLoader

iris = TensorDataset(torch.FloatTensor(X),torch.LongTensor(y)) // y is long tensor in this example of classification for int - could be anything float also

iris_loader = DataLoader(iris, batch_size = 50, shuffle = True)

for i_batch, sample_batch in enumerate(iris_loader):
    print i_batch, sample_batch


### i batch is the serial number of the loaded batch

A simple ANN is given below

# Import all relevant 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split


# Define class
class Model(nn.Module):
    def __init__(self, in_features=4,h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)

    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))
        return x

# Load the iris data and do a train test split
iris = pd.read_csv('../data/iris.csv')
X = iris.drop('target', axis=1)
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Convert all the values to tensors
X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)
y_train = torch.LongTensor(y_train.values)
y_test = torch.LongTensor(y_test.values)
# y_train = F.one_hot(torch.LongTensor(y_train))  # not needed with Cross Entropy Loss
# y_test = F.one_hot(torch.LongTensor(y_test))

# instantiate model, , set loss function, optimization function
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr =0.01)
losses = []
epochs=150
# for each epoch
for i in range(epochs):
    i+=1
    # do forward prop, compute loss and store
    y_pred = model.forward(X_train)
    loss = criterion(y_pred,y_train)
    losses.append(loss.item())
    # set the optimizer gradient to zero
    optimizer.zero_grad()
    # do a back propogation
    loss.backward()
    # update optimizer parameters in next step
    optimizer.step()

# save the model

torch.save(model.state_dict(), 'iris_model.pt')


# plot the results
import matplotlib.pyplot as plt
plt.plot(range(150),losses)

# Load the model
new_model = Model()
new_model.load_state_dict(torch.load('iris_model.pt'))
new_model.eval()

# evaluate the model onthe test set - detach the gradient as no back prop is going to be done here
with torch.no_grad():
    y_pred = new_model.forward(X_test)
    loss = criterion(y_pred,y_test)

for i,data in enumerate(y_pred):
    print(i,data.argmax().item(),y_test[i].item())

# Alternate way to save the entire file and not just the weights
# save the entire model and not just the weights
torch.save(model,'iris.pkl')
# load the pkl file
new_model = torch.load('iris.pkl')
new_model.eval()

# Handling data - categorical and continuous
np.where(df['Hour']>12,'am','pm')
df[cat]=df[cat].astype('category')
df[cat].cat.categories
df[cat].cat.codes


# Hack to get the continuous and categorical columns into a np array
cats = np.stack([df[col].cat.codes.values for col in cat_cols],axis=1)
cats = np.stack([df[col].values for col in cont_cols],axis=1)


Convolutional neural network

Layer types - Convolution layers
Pooling layers - reduce the number of outputs and usually placed between the convolution layers and at the end

Convolution layer characteristics - Number of filters, Color channels(1 gray scale,, 3 color), Kernel size, padding
conv1 = nn.Conv2d(1,6,3) # no padding
Pooling layer : function type max or average , size , stride
If no padding, we lose 2 cells in end
Size of the output o = trunc( [(conv layer-2)/2-2]/2] (for a 2X2 pool with a stride of 2) and 2 convolutional layers

This is the input to the fixed ann at the end - o*o*(no of convolution filters in last layer)


Trained models - Alexnet is trained for image classification
The model has the last layers as adaptive NN layers that can be redone for the classes we have

Ensure No gradient only for the initial convolutional layers when we train to save time

CNN Module Definition
# DON'T WRITE HERE
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)
    
torch.manual_seed(101)
model = ConvolutionalNetwork()

Train the model
# DON'T WRITE HERE
epochs = 5

for i in range(epochs):
    for X_train, y_train in train_loader:

        # Apply the model
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # OPTIONAL print statement
    print(f'{i+1} of {epochs} epochs completed and loss: {loss}')

    Test
    # DON'T WRITE HERE
model.eval()

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_loader:
        y_val = model(X_test)
        predicted = torch.max(y_val,1)[1]
        correct += (predicted == y_test).sum()
        
print(f'Test accuracy: {correct.item()}/{len(test_data)} = {correct.item()*100/(len(test_data)):7.3f}%')


RNN

Input t
Short term mem input C(t-1)
Long Term mem input h(t-1)


Output t
Short term mem output C(t)
Long Term mem output h(t)
In LSTM the last few records are first split into the validation split
The training is then split into batches of X and y starting from the earliest time series record
The output from lstm , the last value is taken

class LSTMnetwork(nn.Module):
    def __init__(self,input_size=1,hidden_size=100,output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Add an LSTM layer:
        self.lstm = nn.LSTM(input_size,hidden_size)
        
        # Add a fully-connected layer:
        self.linear = nn.Linear(hidden_size,output_size)
        
        # Initialize h0 and c0:
        self.hidden = (torch.zeros(1,1,self.hidden_size),
                       torch.zeros(1,1,self.hidden_size))

    def forward(self,seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq),1,-1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq),-1))
        return pred[-1]  # we only want the last value

Remember that the hidden and short term memory should be reset to 0 for running each epoch


In LSTM the last few records are first split into the validation split
The training is then split into batches of X and y starting from the earliest time series record
The output from lstm , the last value is taken
