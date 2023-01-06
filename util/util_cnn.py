import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
# import torchvision
# import torch.nn.functional as F

from util.naming import MNIST_CNN_PATH, device

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
VERSION = 4

class CNNModel(torch.nn.Module):
    def __init__(self, seed=0, cb1_channels=[16], cb2_channels=[32]):
        super(CNNModel, self).__init__()

        torch.random.manual_seed(seed)

        layers = []
        def add_conv(in_channels, out_channels, kernel_size=5, padding=0):
            nonlocal layers
            layers += [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=1),
                       torch.nn.ReLU()]
        
        # Conv Block 1
        channels_in_out = [(1, cb1_channels[0])] + [(cb1_channels[i-1], cb1_channels[i]) for i in range(1, len(cb1_channels))]

        for in_c, out_c in channels_in_out[:-1]: add_conv(in_c, out_c, 3, 1)
        in_c, out_c = channels_in_out[-1];       add_conv(in_c, out_c, 5, 0) # wide conv for last layer in block
        
        # Max pool 1
        layers.append(torch.nn.MaxPool2d(kernel_size=2))
     
        # Convolution 2
        add_conv(in_channels=cb1_channels[-1], out_channels=cb2_channels[0])
        for i in range(1, len(cb2_channels)):
            add_conv(in_channels=cb2_channels[i-1], out_channels=cb2_channels[i], kernel_size=3, padding=1)
        
        # Max pool 2
        layers.append(torch.nn.MaxPool2d(kernel_size=2))

        # Flatten
        layers.append(torch.nn.Flatten())
        
        # Fully connected 1
        layers.append(torch.nn.Linear(cb2_channels[-1] * 4 * 4, 10))

        self.seq = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.seq.forward(x)


def data_loaders(shuffle = True, batch_size = 100):
    # Prepare Dataset
    # load data
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
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = shuffle)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = shuffle)

    return train_loader, test_loader

def first_mnist_batch(batch_size = 100, test=True):
    loaders = data_loaders(shuffle = False, batch_size = batch_size)
    for data, target in loaders[int(test)]:
        return data, target


def train(model, n_iters, train_loader, test_loader):
    # Cross Entropy Loss 
    error = torch.nn.CrossEntropyLoss()

    # SGD Optimizer
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # CNN model training
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []

    while True: # epochs until n_iters reached.
        for i, (images, labels) in enumerate(train_loader):
            
            train = Variable(images.view(100,1,28,28))
            labels = Variable(labels)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward propagation
            outputs = model(train)
            
            # Calculate softmax and ross entropy loss
            loss = error(outputs, labels)
            
            # Calculating gradients
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            count += 1
            
            if count % 50 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    
                    test = Variable(images.view(100,1,28,28))
                    
                    # Forward propagation
                    outputs = model(test)
                    
                    # Get predictions from the maximum value
                    predicted = torch.max(outputs.data, 1)[1]
                    
                    # Total number of labels
                    total += len(labels)
                    
                    correct += (predicted == labels).sum()
                
                accuracy = 100 * correct / float(total)
                
                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            def log(): print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
            if count == n_iters: log(); return model, iteration_list, loss_list, accuracy_list
            if count % 500 == 0: log()

def init_and_train_and_store(params):
    seed, cb1_channels, cb2_channels = params

    train_loader, test_loader = data_loaders()
    print(f"Start training model with seed={seed}, cb1_channels={cb1_channels}, cb2_channels={cb2_channels}.")
    ret =  train(CNNModel(seed, cb1_channels, cb2_channels), n_iters=2500, train_loader=train_loader, test_loader=test_loader)
    
    fn = params_to_filename(seed, cb1_channels, cb2_channels)
    torch.save(ret[0].state_dict(), "./models/"+fn)
    
    print(f"Stored trained model with seed={seed}, cb1_channels={cb1_channels}, cb2_channels={cb2_channels}.")
    return ret

# for loading and storing
def params_to_filename(seed, cb1_channels, cb2_channels):
    a, b = '', ''
    for width in cb1_channels: a += '-' + str(width)
    for width in cb2_channels: b += '-' + str(width)
    return f"mnist_cnn_v4_cb1{a}_cb2{b}_seed-{seed}.torch"

def params_from_filename(fn):
    cb1_channels = fn.split('cb1-')[1].split('_')[0].split('-')
    cb2_channels = fn.split('cb2-')[1].split('_')[0].split('-')
    seed = fn.split('seed-')[1].split('.torch')[0]
    return int(seed), \
           [int(width) for width in cb1_channels], \
           [int(width) for width in cb2_channels]

def load_mnist_v4_models():
    # load v4 models
    model_dict = {}
    for fn in os.listdir(MNIST_CNN_PATH):
        if 'mnist_cnn_v4' in fn:
            params = params_from_filename(fn)
            cnn_model = CNNModel(*params).to(device)
            cnn_model.load_state_dict(torch.load(os.path.join(MNIST_CNN_PATH, fn)))
            model_dict[fn[13:-6]] = cnn_model

    return model_dict