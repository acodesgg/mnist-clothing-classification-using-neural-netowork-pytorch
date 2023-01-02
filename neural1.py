import torch
from torch import nn # neural network 

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 3),# first layer to second layer(hidden layer)
            nn.ReLU(),
            nn.Linear(3, 3), #second (hidden) layer to output layer
        )
     
    #feed forward pass(passing output to another layer)    
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) #apply which is similar to logisitcs regression
        return logits
    
# first , create an instance of NeuralNetwork, and move it to the 
# device , and print its structure

model = NeuralNetwork()
print(model)

X = torch.rand(1,5)
logits = model(X)

# softmax activation converts the logits to probabilities
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax()

print(y_pred) # label output prediction
print(pred_probab.sum()) # softmax activation method
print(pred_probab.argmax()) # same with y_pred

        