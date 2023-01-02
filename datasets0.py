import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST #imported as PIL(python image library)format,need to convert into tensor
from torchvision.transforms import ToTensor , Lambda
import matplotlib.pyplot as plt

BATCH_SIZE = 64

# root is the path where the train/test data is stored
# train specifies training or test dataset to download
# download = True downloads the data from the internet if it's not available at root
# transform and target_transform specify the feature and label transformations

training_data = FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    # ToTensor converts a PIL image or NumPy ndarray to a floatTensor
    # and normalizes the image in the range[0.0,1.0]
    transform = ToTensor(), # transform only to features
    target_transform = Lambda(lambda x: torch.zeros(10,dtype=torch.float).scatter_(dim=0,index=torch.tensor(x),value=1)) # transform only to label/target , for times we need the labels as one-hot encoded tensors
)

test_data = FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(lambda x: torch.zeros(10,dtype=torch.float).scatter_(dim=0,index=torch.tensor(x),value=1))
)

device = "cuda" if torch.cuda.is_available else "cpu"
training_data.data.to(device)
training_data.targets.to(device)

# 10 classes of fashion items
labels_map = {
    0 : "T-shirt/top",
    1 : "Trouser",
    2 : "Pullover",
    3 : "Dress",
    4 : "Coat",
    5 : "Sandal",
    6 : "Shirt",
    7 : "Sneaker",
    8 : "Bag",
    9 : "Ankle boot",
}

figure = plt.figure(figsize = (8,8))
cols,rows = 3,3

for i in range(1,cols * rows + 1):
    sample_idx = torch.randint(len(training_data),size = (1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows,cols,i)
    # modified to account for the one hot encoding , need to change cuz of one hot encoding format
    plt.title(labels_map[list(label).index(1.)])
    plt.axis("off")
    plt.imshow(img.squeeze(),cmap = "gray")
plt.show()

# pass samples in mini batches to reshuffle the data at every time it loops in order to prevent overfitting
# python's multiprocessing

train_dataloader = DataLoader(training_data, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = True)
    
