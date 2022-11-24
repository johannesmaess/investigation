from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
# import torchvision
# import torch.nn.functional as F



# ### DATASET ###

# # Define batch size, batch size is how much data you feed for training in one iteration

# batch_size_train = 64 # We use a small batch size here for training
# batch_size_test = 1024 #

# # define how image transformed
# image_transform = torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ])
# #image datasets
# train_dataset = torchvision.datasets.MNIST('dataset/',
#                                            train=True,
#                                            download=True,
#                                            transform=image_transform)
# test_dataset = torchvision.datasets.MNIST('dataset/',
#                                           train=False,
#                                           download=True,
#                                           transform=image_transform)
# #data loaders
# def get_train_loader(shuffle=True):
#     return torch.utils.data.DataLoader(train_dataset,
#                                            batch_size=batch_size_train,
#                                            shuffle=True)
# train_loader = get_train_loader()

# def get_test_loader(shuffle=True):
#     return torch.utils.data.DataLoader(test_dataset,
#                                           batch_size=batch_size_test,
#                                           shuffle=shuffle)
# test_loader = get_test_loader()



# ### MODEL ###

# # class InfoLayer(nn.Module):
# #     def forward(self, x):
# #         print(x.shape)
# #         return x

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.features = torch.nn.Sequential(
#             nn.Conv2d(1, 10, kernel_size=5, stride=1),
#             nn.MaxPool2d(kernel_size=2),
#             nn.ReLU(),
#             nn.Conv2d(10, 20, kernel_size=5, stride=1),
#             nn.Dropout2d(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.ReLU(),
#         )
#         self.classifier = torch.nn.Sequential(
#             nn.Linear(320, 50),
#             nn.ReLU(),
#             nn.Dropout1d(),
#             nn.Linear(50, 10)
#          )

#     # def put_info_layers(self):
#     #     orddict = self.features._modules
#     #     l=[]
#     #     for ind, layer in orddict.items(): 
#     #         l += [InfoLayer(), layer, ]

#     #     self.features = torch.nn.Sequential(*l)


#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(-1, 320)
#         x = self.classifier(x)
#         return F.log_softmax(x, dim=1)


##define test function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1,1,28,28)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))





### Model from kaggle ###
# kaggle.com/code/kanncaa1/pytorch-tutorial-for-deep-learning-lovers

# Create CNN Model
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = torch.nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = torch.nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = torch.nn.Linear(32 * 4 * 4, 10) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        
        # flatten
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        
        return out


def data_loaders():
    # Prepare Dataset
    # load data
    batch_size = 100
    train = pd.read_csv(r"./dataset/kaggle_input/train.csv",dtype = np.float32)

    # split data into features(pixels) and labels(numbers from 0 to 9)
    targets_numpy = train.label.values
    features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization

    # train test split. Size of train data is 80% and size of test data is 20%. 
    features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                                targets_numpy,
                                                                                test_size = 0.2,
                                                                                random_state = 42) 

    # create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
    featuresTrain = torch.from_numpy(features_train)
    targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

    # create feature and targets tensor for test set.
    featuresTest = torch.from_numpy(features_test)
    targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long


    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
    test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader