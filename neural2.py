import torch
from torch import nn # neural network 
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchviz import make_dot # to visualize computation graph for neural network in pytorch

BATCH_SIZE = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

# 28 x 28 dataset 
training_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

train_dataloader = DataLoader(training_data,batch_size=BATCH_SIZE,shuffle = True)   
train_features, _ = next(iter(train_dataloader))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        # convert each 28 x 28 2d image into a continguous array of 784 pixel values
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            # relu is used instead of sigmoid function cuz relu gives sparse activations so the data is less likely to overfit and faster training
            # relu -> wx+bias <= 0 -> sparse activation in neural network       max(0,x)
            nn.ReLU(),
            nn.Linear(512, 512), 
            nn.ReLU(),
            nn.Linear(512, 10) #(input layer from previous layer, output label count)
        )
      
    #forward pass    
    def forward(self,x):
        #gpu for training
        x = x.to(device)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) #apply which is similar to logisitcs regression
        return logits
    
# first , create an instance of NeuralNetwork, and move it to the 
# device , and print its structure

model = NeuralNetwork().to(device)
print(model)

yhat = model(train_features)    #estimates of y(label)

#visualization
make_dot(yhat,params=dict(model.named_parameters()),show_attrs=True,show_saved=True).render("nn2",format="png")

# X = torch.rand(1,5)
# logits = model(X)
# # softmax activation converts the logits to probabilities
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax()


        