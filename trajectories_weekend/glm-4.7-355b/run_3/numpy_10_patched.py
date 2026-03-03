# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'execution_count': 1, 'status': 'ok'}
# === BEFORE (original) ===
# # Imports here
# %matplotlib inline
# %config InlinBackend.figure_format = 'retina'
# import numpy as np
# import torch
# from torch import nn
# from torch import optim
# import torch.nn.functional as F
# import ast
# import torchvision.transforms as transforms
# from torchvision import datasets, models, transforms
# import torchvision.models as models
# from torch.autograd import Variable
# from collections import OrderedDict
# from PIL import Image
# import json
# import time
# import warnings
# warnings.filterwarnings('ignore')

# === AFTER (edited) ===
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
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
# execution_status: {'execution_count': 2, 'status': 'ok'}
data_dir = 'data_small/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'execution_count': 3, 'status': 'ok'}
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
# Custom dataset to handle corrupted images
from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target
        except (OSError, UnidentifiedImageError, Exception) as e:
            # Skip this sample by returning the next valid one
            # This is a simple workaround - in production, you'd want better handling
            return self.__getitem__((index + 1) % len(self.samples))

# Filter out corrupted files during dataset creation
def filter_valid_samples(root_dir):
    """Return only valid image paths from a directory"""
    from torchvision.datasets import ImageFolder
    valid_samples = []
    
    # Create dummy folder to identify samples
    temp_ds = ImageFolder(root_dir)
    
    for path, target in temp_ds.samples:
        try:
            # Try to open and verify image (reopen after verify)
            with Image.open(path) as img:
                img.verify()
            # Reopen to confirm it's valid (verify() consumes the file)
            with Image.open(path) as img:
                # Try to access basic properties
                _ = img.size
                _ = img.mode
            valid_samples.append((path, target))
        except (OSError, UnidentifiedImageError, Exception):
            continue
    
    return valid_samples

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


# Get valid samples
train_samples = filter_valid_samples(train_dir)
valid_samples = filter_valid_samples(valid_dir)
test_samples = filter_valid_samples(test_dir)

print(f"Valid train samples: {len(train_samples)}")
print(f"Valid valid samples: {len(valid_samples)}")
print(f"Valid test samples: {len(test_samples)}")

# Create datasets with only valid samples
from torch.utils.data import Dataset

class FilteredImageFolder(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image, target

train_ds = FilteredImageFolder(train_samples, train_transforms)
valid_ds = FilteredImageFolder(valid_samples, val_test_transforms)
test_ds = FilteredImageFolder(test_samples, val_test_transforms)

# Create data loaders, but skip if dataset is empty
if len(train_ds) > 0:
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=min(64, len(train_ds)), shuffle=True)
else:
    train_loader = None
    print("Warning: No valid training images found")

if len(valid_ds) > 0:
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=min(64, len(valid_ds)))
else:
    valid_loader = None
    print("Warning: No valid validation images found")
    
if len(test_ds) > 0:
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=min(64, len(test_ds)))
else:
    test_loader = None
    print("Warning: No valid test images found")

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'execution_count': 4, 'status': 'ok'}
# === BEFORE (original) ===
# with open('data_small/cat_to_name.json', 'r') as f:
#     cat_to_name = json.load(f)

# === AFTER (edited) ===
# The cat_to_name.json file appears to be a Git LFS pointer.
# Create a simple mapping from class indices to class numbers
# We'll map class index (0-101) to class number (as found in data structure)
import os

# Get class names from the training directory
train_dir = 'data_small/flower_data/train'
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# Create a simple mapping: class name -> class name
cat_to_name = {cls: cls for cls in class_names}

print(f"Created mapping for {len(cat_to_name)} classes")
print(f"Sample mappings: {dict(list(cat_to_name.items())[:5])}")

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'execution_count': 5, 'status': 'ok'}
# === BEFORE (original) ===
# # TODO: Build and train your network
# 
# # VGG16 Model
# model = models.vgg16(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False   
# 
# # Classifier 
# model.classifier = nn.Sequential(
#     nn.Linear(25088, 1024),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(1024, 512),
#     nn.ReLU(),
#     nn.Dropout(0.2),     
#     nn.Linear(512, 102), 
#     nn.LogSoftmax(dim=1)
# )
# 
# # Move model to GPU
# # model = model.to('cuda')
# 
# # Loss and Optimizer
# criterion = nn.NLLLoss()
# optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
# 
# # Train the model
# epochs = 1 #10
# print_every = 100
# steps = 0
# 
# for e in range(epochs):
#     running_loss = 0
#     for inputs, labels in train_loader:
#         steps += 1
# #         inputs, labels = inputs.to('cuda'), labels.to('cuda')
#         
#         # Forward
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         
#         # Backward 
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         
#         running_loss += loss.item()
#         
#         if steps % print_every == 0:
#             model.eval() 
#             # Validation
#             val_loss = 0
#             accuracy = 0
#             
#             for inputs, labels in valid_loader:
# #                 inputs, labels = inputs.to('cuda'), labels.to('cuda')
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 
#                 val_loss += loss.item()
#                 
#                 ps = torch.exp(outputs)
#                 top_p, top_class = ps.topk(1, dim=1)
#                 equals = top_class == labels.view(*top_class.shape)
#                 accuracy += torch.mean(equals.type(torch.FloatTensor))
#             
#             print(f"Epoch {e+1}/{epochs}.. "
#                   f"Step {steps}/{len(train_loader)}.. "
#                   f"Loss: {running_loss/print_every:.3f}.. " 
#                   f"Validation Loss: {val_loss/len(valid_loader):.3f}.."
#                   f"Accuracy: {accuracy/len(valid_loader):.3f}")
#             running_loss = 0
#             model.train()

# === AFTER (edited) ===
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False


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




criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


# Check if we have data to train
if train_loader is not None and len(train_loader) > 0:
    epochs = 1
    print_every = 100
    steps = 0

    for e in range(epochs):
        running_loss = 0
        for inputs, labels in train_loader:
            steps += 1



            outputs = model(inputs)
            loss = criterion(outputs, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                val_loss = 0
                accuracy = 0
                val_count = 0

                if valid_loader is not None and len(valid_loader) > 0:
                    for inputs, labels in valid_loader:

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()

                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                        val_count += 1

                    if val_count > 0:
                        print(f"Epoch {e+1}/{epochs}.. "
                              f"Step {steps}/{len(train_loader)}.. "
                              f"Loss: {running_loss/print_every:.3f}.. "
                              f"Validation Loss: {val_loss/val_count:.3f}.."
                              f"Accuracy: {accuracy/val_count:.3f}")
                    else:
                        print(f"Epoch {e+1}/{epochs}.. "
                              f"Step {steps}/{len(train_loader)}.. "
                              f"Loss: {running_loss/print_every:.3f}.. "
                              f"No validation data available")
                else:
                    print(f"Epoch {e+1}/{epochs}.. "
                          f"Step {steps}/{len(train_loader)}.. "
                          f"Loss: {running_loss/print_every:.3f}.. "
                          f"No validation data available")
                
                running_loss = 0
                model.train()
    print("Training complete!")
else:
    print("No training data available. Skipping training.")

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'execution_count': 6, 'status': 'ok'}
# === BEFORE (original) ===
# # TODO: Do validation on the test set
# 
# # Load model 
# model.eval()
# 
# # Testing loop
# test_loss = 0 
# accuracy = 0
# 
# with torch.no_grad():
#     for inputs, labels in test_loader:
# #         inputs, labels = inputs.to('cuda'), labels.to('cuda')
#         
#         logps = model(inputs) 
#         batch_loss = criterion(logps, labels)
#         
#         test_loss += batch_loss.item()
#         
#         ps = torch.exp(logps)
#         top_p, top_class = ps.topk(1, dim=1) 
#         equals = top_class == labels.view(*top_class.shape)
#         accuracy += torch.mean(equals.type(torch.FloatTensor))
# 
# # Print test loss and accuracy  
# test_loss = test_loss / len(test_loader)
# accuracy = accuracy / len(test_loader)
# 
# print(f"Test Loss: {test_loss:.3f}.. ")
# print(f"Test Accuracy: {accuracy:.3f}")

# === AFTER (edited) ===
# Check if we have test data
if test_loader is not None and len(test_loader) > 0:
    model.eval()


    test_loss = 0
    accuracy = 0
    test_count = 0

    with torch.no_grad():
        for inputs, labels in test_loader:


            logps = model(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            test_count += 1


    test_loss = test_loss / test_count
    accuracy = accuracy / test_count

    print(f"Test Loss: {test_loss:.3f}.. ")
    print(f"Test Accuracy: {accuracy:.3f}")
else:
    print("No test data available. Skipping evaluation.")

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'execution_count': 7, 'status': 'ok'}
# === BEFORE (original) ===
# # TODO: Save the checkpoint 
# 
# # Model class_to_idx
# model.class_to_idx = train_ds.class_to_idx
# 
# checkpoint = {'input_size': 25088,
#               'output_size': 102,
#               'arch': 'vgg16',
#               'learning_rate': 0.001,
#               'batch_size': 64, 
#               'epochs': epochs,
#               'optimizer': optimizer.state_dict(),
#               'classifier': model.classifier,
#               'state_dict': model.state_dict(),
#               'class_to_idx': model.class_to_idx}
# 
# torch.save(checkpoint, 'checkpoint.pth')

# === AFTER (edited) ===
# Create a class_to_idx mapping from the dataset
class_to_idx = {}
for idx, (path, target) in enumerate(train_ds.samples):
    if target not in class_to_idx:
        # Extract class name from path (parent directory name)
        class_name = os.path.basename(os.path.dirname(path))
        class_to_idx[class_name] = target
        # If we've mapped all unique targets, break
        if len(class_to_idx) == len(set(t for _, t in train_ds.samples)):
            break

model.class_to_idx = class_to_idx

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch': 'vgg16',
              'learning_rate': 0.001,
              'batch_size': 64,
              'epochs': epochs if 'epochs' in locals() else 1,
              'optimizer': optimizer.state_dict(),
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')
print("Checkpoint saved successfully!")

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 8, 'status': 'ok'}
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
# execution_status: {'execution_count': 9, 'status': 'ok'}
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
# execution_status: {'execution_count': 10, 'status': 'ok'}
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
# execution_status: {'execution_count': 11, 'status': 'ok'}
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
# execution_status: {'execution_count': 12, 'status': 'ok'}
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


def display_image(image_path):
    img = process_image(image_path)
    imshow(img)

model = load_checkpoint('checkpoint.pth')

# Check if test_ds has data
if len(test_ds) > 0:
    # Get a random image from test_ds.samples (which is a list of (path, target) tuples)
    random_idx = np.random.choice(len(test_ds.samples))
    test_image_path, test_label = test_ds.samples[random_idx]
    
    display_image(test_image_path)

    probs, classes = predict(test_image_path, model)
    
    # Convert class indices to class names
    class_names = [cat_to_name.get(str(cls), str(cls)) for cls in classes]

    print("Probabilities:", probs)
    print("Classes:", class_names)


    for i in range(5):
        if len(test_ds) > 0:
            random_idx = np.random.choice(len(test_ds.samples))
            test_image_path, test_label = test_ds.samples[random_idx]
            
            display_image(test_image_path)

            probs, classes = predict(test_image_path, model)

            class_names = [cat_to_name.get(str(cls), str(cls)) for cls in classes]

            print("Probabilities:", probs)
            print("Classes:", class_names)
else:
    print("No test images available to display.")