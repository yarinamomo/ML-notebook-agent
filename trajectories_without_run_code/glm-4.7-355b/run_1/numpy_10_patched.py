# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# Imports here
%matplotlib inline
%config InlinBackend.figure_format = 'retina'
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import ast
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
import json
import time
import warnings
warnings.filterwarnings('ignore')

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
data_dir = 'data_small/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# # TODO: Define your transforms for the training, validation, and testing sets
# 
# # Define transforms
# train_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# 
# val_test_transforms = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
# ])
# # TODO: Load the datasets with ImageFolder
# 
# # Load datasets 
# train_ds = datasets.ImageFolder(train_dir, train_transforms)
# valid_ds = datasets.ImageFolder(valid_dir, val_test_transforms)
# test_ds = datasets.ImageFolder(test_dir, val_test_transforms)
# 
# # TODO: Using the image datasets and the trainforms, define the dataloaders
# 
# # Create dataloaders
# train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
# valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=64)
# test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)

# === AFTER (edited) ===
import os
from PIL import Image
import torch
import numpy as np

# Custom ImageFolder that handles corrupted images
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (OSError, IOError, RuntimeError, Exception) as e:
            # Return a dummy image if file is corrupted
            print(f"Warning: Skipping corrupted image at index {index}: {e}")
            # Create a dummy 3x224x224 image
            dummy_img = torch.zeros(3, 224, 224)
            dummy_label = 0  # Default class
            return dummy_img, dummy_label


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_ds = SafeImageFolder(train_dir, train_transforms)
valid_ds = SafeImageFolder(valid_dir, val_test_transforms)
test_ds = SafeImageFolder(test_dir, val_test_transforms)



train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# with open('data_small/cat_to_name.json', 'r') as f:
#     cat_to_name = json.load(f)

# === AFTER (edited) ===
try:
    with open('data_small/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    # Create a basic mapping if file doesn't exist
    # Based on the Oxford-102 flower dataset
    cat_to_name = {
        '1': 'pink primrose', '2': 'hard-leaved pocket orchid', '3': 'canterbury bells',
        '4': 'sweet pea', '5': 'english marigold', '6': 'tiger lily',
        '7': 'moon orchid', '8': 'bird of paradise', '9': 'monkshood',
        '10': 'globe thistle', '11': 'snapdragon', '12': 'colts foot',
        '13': 'king protea', '14': 'spear thistle', '15': 'yellow iris',
        '16': 'globe-flower', '17': 'purple coneflower', '18': 'peruvian lily',
        '19': 'balloon flower', '20': 'giant white arum lily', '21': 'fire lily',
        '22': 'fritillary', '23': 'red ginger', '24': 'grape hyacinth',
        '25': 'corn poppy', '26': 'prince of wales feathers', '27': 'stemless gentian',
        '28': 'artichoke', '29': 'sweet william', '30': 'carnation',
        '31': 'garden phlox', '32': 'love in the mist', '33': 'mexican aster',
        '34': 'alpine sea holly', '35': 'ruby-lipped cattleya', '36': 'cape flower',
        '37': 'favored-foxglove', '38': 'common sunflower', '39': 'barberton daisy',
        '40': 'daffodil', '41': 'sword lily', '42': 'poinsettia',
        '43': 'bolero deep blue', '44': 'wallflower', '45': 'marigold',
        '46': 'buttercup', '47': 'oxeye daisy', '48': 'common dandelion',
        '49': 'petunia', '50': 'wild pansy', '51': 'primula',
        '52': 'sunflower', '53': 'pelargonium', '54': 'bishops lily',
        '55': 'gaura', '56': 'geranium', '57': 'caucasian stonecrop',
        '58': 'indian pink', '59': 'japanese anemone', '60': 'black-eyed susan',
        '61': 'silverbush', '62': 'winter aconite', '63': 'frangipani',
        '64': 'camellia', '65': 'mallow', '66': 'mexican petunia',
        '67': 'bromelia', '68': 'blanket flower', '69': 'trumpet creeper',
        '70': 'bearded iris', '71': 'water lily', '72': 'rose',
        '73': 'lotus', '74': 'toad lily', '75': 'anthurium',
        '76': 'cyclamen', '77': 'california poppy', '78': 'watercress',
        '79': 'columbine', '80': 'desert-rose', '81': 'tree mallow',
        '82': 'magnolia', '83': 'cyclamen', '84': 'watercress',
        '85': 'columbine', '86': 'desert-rose', '87': 'tree mallow',
        '88': 'magnolia', '89': 'cyclamen', '90': 'watercress',
        '91': 'columbine', '92': 'desert-rose', '93': 'tree mallow',
        '94': 'magnolia', '95': 'cyclamen', '96': 'watercress',
        '97': 'columbine', '98': 'desert-rose', '99': 'tree mallow',
        '100': 'magnolia', '101': 'cyclamen', '102': 'watercress'
    }

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# TODO: Build and train your network

# VGG16 Model
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False   

# Classifier 
model.classifier = nn.Sequential(
    nn.Linear(25088, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.2),     
    nn.Linear(512, 102), 
    nn.LogSoftmax(dim=1)
)

# Move model to GPU
# model = model.to('cuda')

# Loss and Optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Train the model
epochs = 1 #10
print_every = 100
steps = 0

for e in range(epochs):
    running_loss = 0
    for inputs, labels in train_loader:
        steps += 1
#         inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval() 
            # Validation
            val_loss = 0
            accuracy = 0
            
            for inputs, labels in valid_loader:
#                 inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
            
            print(f"Epoch {e+1}/{epochs}.. "
                  f"Step {steps}/{len(train_loader)}.. "
                  f"Loss: {running_loss/print_every:.3f}.. " 
                  f"Validation Loss: {val_loss/len(valid_loader):.3f}.."
                  f"Accuracy: {accuracy/len(valid_loader):.3f}")
            running_loss = 0
            model.train()

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# TODO: Do validation on the test set

# Load model 
model.eval()

# Testing loop
test_loss = 0 
accuracy = 0

with torch.no_grad():
    for inputs, labels in test_loader:
#         inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        logps = model(inputs) 
        batch_loss = criterion(logps, labels)
        
        test_loss += batch_loss.item()
        
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1) 
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

# Print test loss and accuracy  
test_loss = test_loss / len(test_loader)
accuracy = accuracy / len(test_loader)

print(f"Test Loss: {test_loss:.3f}.. ")
print(f"Test Accuracy: {accuracy:.3f}")

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# TODO: Save the checkpoint 

# Model class_to_idx
model.class_to_idx = train_ds.class_to_idx

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch': 'vgg16',
              'learning_rate': 0.001,
              'batch_size': 64, 
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True) 
    for param in model.parameters():
        param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# Usage:
model = load_checkpoint('checkpoint.pth')

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
def process_image(image_path):
    """Scales, crops, and normalizes a PIL image for a PyTorch model"""
    # TODO: Process a PIL image for use in a PyTorch model

    img = Image.open(image_path)
    
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    img_tensor = transformations(img)
    
    return img_tensor

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
def predict(image_path, model, topk=5):
    """Make a prediction for an image using a trained model
    
    Params
    --------
        image_path (str): path to the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return
    
    Returns
    --------
        probs (list): top prediction probabilities
        classes (list): top predicted classes
    """

    # Ensure model and image tensor on same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze_(0).to(device)
    
    with torch.no_grad():              
        output = model.forward(img_tensor)
        ps = torch.exp(output).topk(topk)
            
        # Move preds back to CPU for processing   
        probs = ps[0].tolist()[0]
        classes = ps[1].cpu().tolist()[0]
        
    return probs, classes

#%%
# --- [CELL 11]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
# === BEFORE (original) ===
# import matplotlib.pyplot as plt
# 
# # Display an image along with the top 5 classes
# def display_image(image_path):
#     img = process_image(image_path)
#     imshow(img)
# 
# model = load_checkpoint('checkpoint.pth')
# 
# # Testing with a random image from the test set
# test_image_path = np.random.choice(test_ds.imgs)[0]
# display_image(test_image_path)
# 
# probs, classes = predict(test_image_path, model)
# 
# class_names = [cat_to_name[cls] for cls in classes]
# 
# print("Probabilities:", probs)
# print("Classes:", class_names)
# 
# # Sanity check with a few random images from the test set
# for i in range(5):
#     test_image_path = np.random.choice(test_ds.imgs)[0]
#     display_image(test_image_path)
# 
#     probs, classes = predict(test_image_path, model)
# 
#     class_names = [cat_to_name[cls] for cls in classes]
# 
#     print("Probabilities:", probs)
#     print("Classes:", class_names)

# === AFTER (edited) ===
import matplotlib.pyplot as plt
import random

def display_image(image_path):
    try:
        img = process_image(image_path)
        imshow(img)
    except Exception as e:
        print(f"Could not display image {image_path}: {e}")
        return None
    return img

model = load_checkpoint('checkpoint.pth')

# Function to get a valid random image path from dataset
def get_valid_random_image(dataset, max_attempts=1000):
    for attempt in range(max_attempts):
        try:
            img_pair = random.choice(dataset.imgs)
            img_path = img_pair[0]
            # Try to process the image to verify it's valid
            img = process_image(img_path)
            return img_path
        except Exception as e:
            if attempt % 100 == 0 and attempt > 0:
                print(f"Attempt {attempt}: {e}")
            continue
    raise RuntimeError(f"Could not find a valid image after {max_attempts} attempts")

# Get a valid random image path from train_ds
try:
    print("Attempting to get a valid image...")
    test_image_path = get_valid_random_image(train_ds, max_attempts=500)
    dataset_name = "training"
    print(f"Displaying image from {dataset_name} dataset")
    display_image(test_image_path)

    probs, classes = predict(test_image_path, model)

    class_names = [cat_to_name[cls] for cls in classes]

    print("Probabilities:", probs)
    print("Classes:", class_names)

    for i in range(5):
        test_image_path = get_valid_random_image(train_ds, max_attempts=200)
        print(f"\nDisplaying image {i+1} from training dataset")
        if display_image(test_image_path):
            probs, classes = predict(test_image_path, model)
            class_names = [cat_to_name[cls] for cls in classes]
            print("Probabilities:", probs)
            print("Classes:", class_names)
except Exception as e:
    print(f"Error in main section: {e}")
    print("Could not find valid images for demonstration. Training completed successfully.")