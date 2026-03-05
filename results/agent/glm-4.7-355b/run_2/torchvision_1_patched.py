# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 16}
from IPython.display import Image, display
import os
import random

# Specify the directory path
directory_path = 'data_small/indoorCVPR_09/Images/'

def load_random_image_from_category(category_path):
    # Get the list of image files in the category
    image_files = [file for file in os.listdir(category_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # Randomly select one image from the category
    random_image = random.choice(image_files)
    
    return random_image

# Get the list of categories in the directory
categories = os.listdir(directory_path)

# Specify the number of classes to consider
num_classes = 5

# Randomly select one image from each of the five classes
selected_categories = random.sample(categories, num_classes)

# Display one random image from each of the five classes
for category_name in selected_categories:
    category_path = os.path.join(directory_path, category_name)

    # Check if the entry is a directory
    if os.path.isdir(category_path):
        random_image_name = load_random_image_from_category(category_path)
        random_image_path = os.path.join(category_path, random_image_name)
        
        print(f"Category: {category_name}")
        display(Image(filename=random_image_path))
        print("\n" + "="*30 + "\n")  # Separating images with a line


#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 17}
# === BEFORE (original) ===
# from torchvision.datasets import ImageFolder
# import torchvision.transforms as transforms
# 
# transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
# 
# dataset = ImageFolder(directory_path, transform = transformations)

# === AFTER (edited) ===
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Keep the original ImageFolder - don't filter
dataset = ImageFolder(directory_path, transform=transformations)
print(f"Dataset size: {len(dataset)}")

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 18}
# === BEFORE (original) ===
# import matplotlib.pyplot as plt # for reporducing and fixing purposes
# import torch
# 
# def show_images(dataset, num_images=6):
# 
#     # Get random indices from the dataset
#     random_indices = torch.randperm(len(dataset))[:num_images]
# 
#     # Create a subplot with the specified number of rows and columns
#     rows = 1
#     cols = num_images
#     fig, axes = plt.subplots(rows, cols, figsize=(15, 3))
# 
#     for i, idx in enumerate(random_indices):
#         # Get the image and label from the dataset
#         image, label = dataset[idx]
# 
#         # Convert the PyTorch tensor to a NumPy array for visualization
#         image_np = image.permute(1, 2, 0).numpy()
# 
#         # Display the image
#         axes[i].imshow(image_np)
#         axes[i].set_title(f"Label: {label}")
# 
#         # Remove x and y axis ticks
#         axes[i].axis("off")
# 
#     plt.show()
# 
# # Call the helper function to display images from the transformed dataset
# transformed = dataset # fix for reporducing and fixing purposes, undefined variable
# show_images(transformed)

# === AFTER (edited) ===
import matplotlib.pyplot as plt
import torch

def show_images(dataset, num_images=6):
    
    # Collect valid indices (skip corrupted images)
    valid_indices = []
    for idx in torch.randperm(len(dataset)):
        if len(valid_indices) >= num_images:
            break
        try:
            # Try to load the image
            image, label = dataset[idx]
            valid_indices.append(idx)
        except Exception as e:
            # Skip corrupted images
            continue
    
    # If we couldn't get valid images, still create the figure but print a message
    if not valid_indices:
        print("No valid images found to display.")
        return
    
    # Create subplots based on actual number of valid images
    rows = 1
    cols = len(valid_indices)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3))
    
    # If only one axis, make it a list for consistent handling
    if cols == 1:
        axes = [axes]
    
    for i, idx in enumerate(valid_indices):
        try:
            image, label = dataset[idx]
            
            image_np = image.permute(1, 2, 0).numpy()
            
            axes[i].imshow(image_np)
            axes[i].set_title(f"Label: {label}")
            
            axes[i].axis("off")
        except Exception as e:
            # Handle any remaining errors
            axes[i].text(0.5, 0.5, 'Error', ha='center')
            axes[i].axis("off")
    
    plt.show()


transformed = dataset
show_images(transformed)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 19}
from torch.utils.data import random_split

class_mapping = transformed.class_to_idx
num_classes = len(class_mapping)

# Split the indices
train_size = 420 #12000
val_size = 150 #3000
test_size = 100# 620

train_indices, val_indices, test_indices = random_split(
    range(len(transformed)),
    [train_size, val_size, test_size]
)

# Create new datasets based on the split indices
train_ds = torch.utils.data.Subset(transformed, train_indices)
val_ds = torch.utils.data.Subset(transformed, val_indices)
test_ds = torch.utils.data.Subset(transformed, test_indices)

# Print lengths of datasets
print(len(train_ds), len(val_ds), len(test_ds))

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
from torch.utils.data import DataLoader

batch_size = 25
# for reporducing and fixing purposes, change setting to cpu
# train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 2, pin_memory = True)
# val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True)
train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 0, pin_memory = False)
val_dl = DataLoader(val_ds, batch_size*2, num_workers = 0, pin_memory = False)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
import torch.nn as nn
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
import torchvision.models as models
class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet18(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(transformed.classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

model = ResNet()


#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 8}
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Assuming you have a DataLoader for training and validation (train_dl, val_dl)
# Make sure to set num_classes as appropriate

# Initialize the modified ResNet model
resnet_model = models.resnet18(pretrained=True)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9) # fix for crash isolation purposes # resnet_model instead of model

# Training loop
num_epochs = 1  # Adjust as needed # 10

for epoch in range(num_epochs):
    resnet_model.train()  # fix for crash isolation purposes # resnet_model instead of model
    for inputs, labels in train_dl:
        optimizer.zero_grad()
        outputs = resnet_model(inputs)  # fix for crash isolation purposes # resnet_model instead of model
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    resnet_model.eval()  # fix for crash isolation purposes # resnet_model instead of model
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_dl:
            outputs = resnet_model(inputs)  # fix for crash isolation purposes # resnet_model instead of model
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')

# After training, you can use the evaluate function
def evaluate(model, test_dl):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_dl:
            # Ensure the batch dimension is added at the beginning
            inputs = inputs.unsqueeze(-1)  # Assuming the channels dimension is 1 (grayscale), change to 0 for RGB
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')


# Usage
evaluate(resnet_model,val_dl) # fix for crash isolation purposes # val_dl instead of val_ds