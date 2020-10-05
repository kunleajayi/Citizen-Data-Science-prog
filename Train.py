import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import sys
import os
from PIL import Image
import torchvision
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
from collections import OrderedDict
from torch.autograd import Variable
import json
import argparse




parser = argparse.ArgumentParser(description='Train Image Classifier')


# Basic usage: python train.py data_dir
parser.add_argument('data_dir', nargs='*', action='store',type = str,
                    default = 'flowers',
                    help='Set directory to load training data, e.g., "flowers"')

# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
parser.add_argument('--save_dir', action='store',type = str,
                    default = 'checkpoint.pth',
                    dest='save_dir',
                    help='Set directory to save checkpoints, e.g., "assets"')


# Choose architecture: python train.py data_dir --arch "vgg13"
parser.add_argument('--arch', action='store',type = str,
                    default = 'VGG16',
                    dest='arch',
                    help='Choose architecture, e.g., "VGG16 or alexnet"')

# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
parser.add_argument('--lr', action='store',type = float,
                    default = 0.001,
                    dest='lr',
                    help='Choose architecture learning rate, e.g., 0.01')


parser.add_argument('--hidden_units', action='store',type = int,
                    default = 1024,
                    dest='hidden_units',
                    help='Choose architecture hidden units, e.g., 1024')



parser.add_argument('--epochs', action='store',type = int,
                    default = 5,
                    dest='epochs',
                    help='Enter number of epochs, e.g. 4')


# Use GPU for training: python train.py data_dir --gpu
parser.add_argument('--GPU', action='store_true',
                    default=False,
                    dest='GPU',
                    help='Use GPU for training, set a switch to true')



arguments = parser.parse_args()



hyperparameters = { 'data_dir': arguments.data_dir,
                    'save_dir': arguments.save_dir,
                    'arch': arguments.arch,
                    'lr': arguments.lr,
                    'hidden_units': arguments.hidden_units,
                    'epochs': arguments.epochs,
                    'device': arguments.GPU}


# Image data directories
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                     ])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                     ])

test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])
                                     ])


# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data  = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the transforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
test_loader  = torch.utils.data.DataLoader(test_data, batch_size=32)

image_datasets = [train_data, valid_data, test_data]
dataloaders = [train_loader, valid_loader, test_loader]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device

# Mapping flowers category label to category name

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print("{} flower category labels mapped to  category names in the  list below:".format(len(cat_to_name)))
cat_to_name



#3. Build the Network
def get_model(model_arch):
    #load a pretrained model
    if (model_arch == 'vgg16'):
        model = models.vgg16(pretrained = True)
    elif (model_arch == 'alexnet'):
        model = models.alexnet(pretrained = True)
    else:
        print("{} selected model not in the option.pls pick either of  vgg16 or alexnet".format(arch))
    return model


def build_model(model, model_arch,hidden_units):
   
    for param in model.parameters():
        param.requires_grad = False

    if (model_arch == 'vgg16'):
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, hyperparameters['hidden_units'])),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(p=0.5)),
                                  ('fc2', nn.Linear(hyperparameters['hidden_units'], 102)),
                                  ('output', nn.LogSoftmax(dim=1))                             
                                  ]))
    elif (model_arch == 'alexnet'):
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(9216, hyperparameters['hidden_units'])),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(p=0.5)),
                                  ('fc2', nn.Linear(hyperparameters['hidden_units'], 102)),
                                  ('output', nn.LogSoftmax(dim=1))                             
                                  ]))
    else:
        print('Error detected')
    return classifier

model = get_model(hyperparameters['arch'].lower())
classifier = build_model(model, hyperparameters['arch'].lower(), hyperparameters['hidden_units'])
model.classifier = classifier
print('\nprinting the selected architecture ' + hyperparameters['arch'] + ' classifier = ')
print(model.classifier)


# Define the loss function
criterion = nn.NLLLoss()


# Define weights optimizer (backpropagation with gradient descent)
# Only train the classifier parameters, feature parameters are frozen
# Set the learning rate
optimizer = optim.Adam(model.classifier.parameters(), hyperparameters['lr'])


# Move the network and data to GPU or CPU
model.to(device)




# Function for the validation pass
def validation(model, valid_loader, criterion):
    
    val_loss = 0
    accuracy = 0
    
    for images, labels in iter(valid_loader):

        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        probabilities = torch.exp(output)
        
        # Class with the highest probability is our predicted class
        equality = (labels.data == probabilities.max(dim=1)[1])
        
        # Accuracy is number of correct predictions divided by all predictions
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return val_loss, accuracy



# Train the classifier
start = time.time()

print('Start training the classifier ')

# loop over the dataset multiple times
epochs = hyperparameters['epochs']  
print_every = 40
steps = 0
for epoch in range(epochs):
    running_loss = 0
    for images, labels in iter(train_loader):
        
        steps += 1

        images, labels = images.to(device), labels.to(device)
      
     
        # Clear the gradients as gradients are accumulated
        optimizer.zero_grad()
        
        #forwad pass
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        
        #Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # printing statistics
        if steps % print_every == 0:
           
        # evaluate model:
            model.eval()
            
              # Turn off gradients for validation,thereby  saving memory and computations
            with torch.no_grad():
                # Validate model
                validation_loss, accuracy = validation(model, valid_loader, criterion)                                               
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(validation_loss/len(valid_loader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))

            
            running_loss = 0
        
        # Return to model training 
            model.train()
            
print('Finished Training the Classifier')

training_duration = time.time() - start

print("\nTotal time: {:.0f}m {:.0f}s".format(training_duration//60, training_duration % 60))
    
    
  # validation on the test set
def Test_Accuracy(model, test_loader):
    # This is aimed at notifying all  layers that we are in eval mode, that way,
    # batchnorm or dropout layers will work in eval mode instead of training mode
    model.eval()
    
      
    # This  impacts the autograd engine and deactivate it
    
    with torch.no_grad():
        
        accuracy = 0
        
        for images, labels in iter(test_loader):

            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
        
            probabilities = torch.exp(output)
        
            # Class with the highest probability is our predicted class
            equality = (labels.data == probabilities.max(dim=1)[1])
        
            # Accuracy is number of correct predictions divided by all predictions
            accuracy += equality.type(torch.FloatTensor).mean()
        
        print("Test Accuracy: {}".format(accuracy/len(test_loader))) 
    
    return Test_Accuracy

Test_Accuracy(model, test_loader)



# Save the checkpoint 
model.class_to_idx = train_data.class_to_idx
checkpoint = {
    'model': model,
    'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_to_idx': train_data.class_to_idx
}
torch.save(checkpoint, 'checkpoint.pth')
