import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchviz import make_dot

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10 # increase this to something reasonable

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device...")

# requirements
# [x] 1.data
# [x] 2.model (neural network)
# [x] 3.loss function
# [x] 4.optimizer/optimization

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)

train_dataloader = DataLoader(training_data,batch_size=BATCH_SIZE,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)

for X,y in test_dataloader:
    print(f"Shape of X[N,C,H,W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # convert 28x28 2d image into a continguous array of 784 pixels
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),            
            nn.Linear(512, 10)
        )
        
    def forward(self,x):
        # gpu for training, no gpu then cpu
        x = x.to(device)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
# optimization with stochastic gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,momentum=0.95)


# requirements
# [x] 1.data
# [x] 2.model (neural network)
# [x] 3.loss function
# [x] 4.optimizer/optimization
def train_loop(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    
    for batch, (X,y) in enumerate(dataloader):
        
        # move X and y in cuda
        X = X.to(device)
        y = y.to(device)
        
        # settign delta W and delta b to zero
        optimizer.zero_grad()
        
        # forward + backward + optimize combo
        # forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
        # for every increment of 100 (64 in a batch * 100 = 6400 samples), print the loss
        if batch % 100 == 0:
            print(f"loss: {loss.item() :> 5f} -- [{batch * len(X)}/{size}]")
            

def test_loop(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss , correct = 0 , 0
    
    with torch.no_grad(): # to reduce memory comsumption
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            # forward (only for inference)
            pred = model(X)
            
            # test_loss explanation
            # 3 minibatches each have 333 images
            # test accuracy/loss for first 333 -> 11 (0+11)
            # test accuracy/loss for second 333 -> 9 (11+9)
            # test accuracy/loss for third 333 -> 15 (20+15)
            # total loss : 35 out of 1000 so that we can calculate accuracy
            test_loss += loss_fn(pred,y)
            
            # pred: [0,0.2,0.25,0.15,0.4]
            correct += (pred.argmax(1)  == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size        
    print(f"Test Error: \n Accuracy: {100 * correct:>0.1f}%, Avg Loss: {test_loss:>5f} \n")    


# loop for training(each loop is called epoch, one epoch -> data,model,loss function,optimization)
for t in range(EPOCHS):
    # train the model with an optimization loop
    # each iteration of the loop is an epoch
    # each epoch consisits of two parts, the training and testing
    
    # iterate over training data and try converging to optimal parameters
    train_loop(train_dataloader,model,loss_fn,optimizer)
    # iterate over testing data and check accuracy to see if the performance is improving
    test_loop(test_dataloader,model,loss_fn)
    
    # on last loop of train-test cycle, produce the computational graph
    if t == EPOCHS -1:
        make_dot(model(X),params=dict(model.named_parameters())).render("Complete Example",format="png")
print(f"Done!")       

# save the model
torch.save(model.state_dict(),"model.pth") 

# loading the model and testing with real data
# model = NeuralNetwork()
# model.load_state_dict(torch.load("model.pth"))
# X,y = test_data[0][0],test_data[0][1]
# model.to(device)
# with torch.no_grad():
#     X = X.to(device)
#     y = y.to(device)
#     pred = model(X)
#     pred[0] # predicted labels
#     pred[0].argmax(0) # predicted label with max probability
#     y # pred[0].argmax(0) and y have the same label so it's correct!