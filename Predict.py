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


parser = argparse.ArgumentParser(description='predict image')

# Command line arguments

# Basic usage: python predict.py /path/to/image checkpoint
parser.add_argument('image_path',nargs='*', action='store',type = str,
                    default = 'flowers/test/15/image_06351.jpg',
                    help='Path to image, e.g., "flowers/test/1/image_06743.jpg"')

parser.add_argument('--checkpoint', action='store',type = str,
                    default = 'checkpoint.pth',
                    help='Directory of saved checkpoints')

# Return top K most likely classes: python predict.py input checkpoint --top_k 3
parser.add_argument('--topk', action='store',type = int,
                    default = 5,
                    dest='topk',
                    help='Return top K most likely classes, e.g., 5')

# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
parser.add_argument('--category_names', action='store',type = str,
                    default = 'cat_to_name.json',
                    dest='category_names',
                    help='mapping of flower categories to real names, e.g., "cat_to_name.json"')

# Use GPU for inference: python predict.py input checkpoint --gpu
parser.add_argument('--GPU', action='store_true',default=False,dest='GPU',
                    help='Use GPU for inference')

arguments = parser.parse_args()


hyperparameters = { 'image_dir': arguments.image_path,
                    'checkpoint_dir': arguments.checkpoint,
                    'category_dir':arguments.category_names,
                    'topk': arguments.topk,
                    'device': arguments.GPU}


# Label mapping

cat_file_name = hyperparameters['category_dir']
with open(cat_file_name, 'r') as f:
    cat_to_name = json.load(f)
print("{} flower category labels mapped to  category names in the  list below:".format(len(cat_to_name)))
cat_to_name

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model

model = load_checkpoint(hyperparameters['checkpoint_dir'])
device = torch.device('cpu')
print(model)


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    
    pil_image = Image.open(image_path)
    
    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio. width is pil_image.size[0]
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((10000, 256))
    else:
        pil_image.thumbnail((256, 10000))
        
    # Crop 
    left_margin = (pil_image.width-224)/2
    bottom_margin = (pil_image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Normalize
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
    # reorder dimensions using ndarray.transpose. 
    # The color channel needs to be first and retain the order of the other two dimensions.
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image


# Implement the code to predict the class from an image file

def predict(model,image_path,topk = 5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
        # Implement the code to predict the class from an image file
    if torch.cuda.is_available() and hyperparameters['device']== 'GPU':
        model.to('cuda:0')
    image = process_image(image_path)

    
    # Convert image to PyTorch tensor first
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    print(image.shape)
    print(type(image))
    
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)
    
    output = model.forward(image)
    
    probabilities = torch.exp(output)
    
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(hyperparameters ['topk'])
    
    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    #print(idx_to_class)
    
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes    

probs, classes = predict('flowers/test/15/image_06351.jpg',model)  
flower_names = [cat_to_name[i] for i in classes]
print(probs)
print(classes)
print(flower_names)
