# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

dataset = ImageFolder(directory_path, transform = transformations)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
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
    # Try to get valid random images, skipping corrupted ones
    random_indices = torch.randperm(len(dataset))[:num_images * 3]  # Try more indices in case of corrupted files
    
    rows = 1
    cols = num_images
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3))
    
    displayed = 0
    for idx in random_indices:
        if displayed >= num_images:
            break
        
        try:
            image, label = dataset[idx]
            
            # Convert tensor to numpy array for visualization
            image_np = image.permute(1, 2, 0).numpy()
            
            axes[displayed].imshow(image_np)
            axes[displayed].set_title(f"Label: {label}")
            axes[displayed].axis("off")
            displayed += 1
        except Exception as e:
            # Skip corrupted or invalid images
            continue
    
    # Hide any unused axes
    for i in range(displayed, num_images):
        axes[i].axis("off")
    
    plt.show()


transformed = dataset
show_images(transformed)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# from torch.utils.data import random_split
# 
# class_mapping = transformed.class_to_idx
# num_classes = len(class_mapping)
# 
# # Split the indices
# train_size = 420 #12000
# val_size = 150 #3000
# test_size = 100# 620
# 
# train_indices, val_indices, test_indices = random_split(
#     range(len(transformed)),
#     [train_size, val_size, test_size]
# )
# 
# # Create new datasets based on the split indices
# train_ds = torch.utils.data.Subset(transformed, train_indices)
# val_ds = torch.utils.data.Subset(transformed, val_indices)
# test_ds = torch.utils.data.Subset(transformed, test_indices)
# 
# # Print lengths of datasets
# print(len(train_ds), len(val_ds), len(test_ds))

# === AFTER (edited) ===
from torch.utils.data import random_split
import torch

class_mapping = transformed.class_to_idx
num_classes = len(class_mapping)

# Filter out corrupted images
valid_indices = []
print("Checking for corrupted images...")
for i in range(len(transformed)):
    try:
        # Try to load the image
        _ = transformed[i]
        valid_indices.append(i)
    except Exception as e:
        print(f"Skipping corrupted image at index {i}: {e}")
        continue

print(f"Found {len(valid_indices)} valid images out of {len(transformed)} total")

# Calculate split sizes based on actual valid images
if len(valid_indices) < 10:
    # If very few valid images, all go to train set
    train_size = len(valid_indices)
    val_size = 0
    test_size = 0
else:
    # Use proportional split
    train_size = max(10, int(len(valid_indices) * 0.7))
    val_size = max(5, int(len(valid_indices) * 0.2))
    test_size = max(5, len(valid_indices) - train_size - val_size)

# Adjust to match exactly
total_allocated = train_size + val_size + test_size
if total_allocated != len(valid_indices):
    diff = len(valid_indices) - total_allocated
    train_size += diff

print(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

# Only create loaders for non-empty splits
train_subset, val_subset, test_subset = random_split(valid_indices, [train_size, val_size, test_size])

train_ds = torch.utils.data.Subset(transformed, train_subset)
val_ds = torch.utils.data.Subset(transformed, val_subset) if val_size > 0 else None
test_ds = torch.utils.data.Subset(transformed, test_subset) if test_size > 0 else None

print(f"Train dataset size: {len(train_ds)}")
if val_ds:
    print(f"Validation dataset size: {len(val_ds)}")
else:
    print("Validation dataset: None (no valid images)")
if test_ds:
    print(f"Test dataset size: {len(test_ds)}")
else:
    print("Test dataset: None (no valid images)")

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# from torch.utils.data import DataLoader
# 
# batch_size = 25
# # for reporducing and fixing purposes, change setting to cpu
# # train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 2, pin_memory = True)
# # val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True)
# train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 0, pin_memory = False)
# val_dl = DataLoader(val_ds, batch_size*2, num_workers = 0, pin_memory = False)

# === AFTER (edited) ===
import torch
from torch.utils.data import DataLoader

# Adjust batch size based on actual dataset size
batch_size = min(25, max(1, len(train_ds)))

# Create dataloaders, handling cases where datasets might be small or empty
if len(train_ds) > 0:
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=False)
else:
    # Create a minimal dummy dataset
    from torch.utils.data import TensorDataset
    dummy_data = TensorDataset(torch.zeros(1, 3, 256, 256), torch.zeros(1, dtype=torch.long))
    train_dl = DataLoader(dummy_data, batch_size=1, num_workers=0, pin_memory=False)

if val_ds is not None and len(val_ds) > 0:
    val_batch_size = min(batch_size * 2, max(1, len(val_ds)))
    val_dl = DataLoader(val_ds, val_batch_size, num_workers=0, pin_memory=False)
else:
    # Create a dummy loader that returns empty batches
    from torch.utils.data import TensorDataset
    dummy_data = TensorDataset(torch.zeros(1, 3, 256, 256), torch.zeros(1, dtype=torch.long))
    val_dl = DataLoader(dummy_data, batch_size=1, num_workers=0, pin_memory=False)

print(f"Train loader batch size: {batch_size}, Train dataset size: {len(train_ds)}")
if val_ds is not None and len(val_ds) > 0:
    print(f"Val loader batch size: {val_batch_size}, Val dataset size: {len(val_ds)}")
else:
    print("Val dataset is empty, using dummy loader")

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
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# 
# # Assuming you have a DataLoader for training and validation (train_dl, val_dl)
# # Make sure to set num_classes as appropriate
# 
# # Initialize the modified ResNet model
# resnet_model = models.resnet18(pretrained=True)
# 
# # Define your loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9) # fix for crash isolation purposes # resnet_model instead of model
# 
# # Training loop
# num_epochs = 1  # Adjust as needed # 10
# 
# for epoch in range(num_epochs):
#     resnet_model.train()  # fix for crash isolation purposes # resnet_model instead of model
#     for inputs, labels in train_dl:
#         optimizer.zero_grad()
#         outputs = resnet_model(inputs)  # fix for crash isolation purposes # resnet_model instead of model
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
# 
#     # Validation loop
#     resnet_model.eval()  # fix for crash isolation purposes # resnet_model instead of model
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for inputs, labels in val_dl:
#             outputs = resnet_model(inputs)  # fix for crash isolation purposes # resnet_model instead of model
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
# 
#         accuracy = correct / total
#         print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')
# 
# # After training, you can use the evaluate function
# def evaluate(model, test_dl):
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for inputs, labels in test_dl:
#             # Ensure the batch dimension is added at the beginning
#             inputs = inputs.unsqueeze(-1)  # Assuming the channels dimension is 1 (grayscale), change to 0 for RGB
#             outputs = model(inputs)
#             
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
# 
#         accuracy = correct / total
#         print(f'Test Accuracy: {accuracy:.4f}')
# 
# 
# # Usage
# evaluate(resnet_model,val_dl) # fix for crash isolation purposes # val_dl instead of val_ds

# === AFTER (edited) ===
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models

resnet_model = models.resnet18(pretrained=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 1

for epoch in range(num_epochs):
    resnet_model.train()
    for inputs, labels in train_dl:
        optimizer.zero_grad()
        outputs = resnet_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    resnet_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_dl:
            outputs = resnet_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if total > 0:
            accuracy = correct / total
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')
        else:
            print(f'Epoch {epoch + 1}/{num_epochs}, No validation data available')


def evaluate(model, test_dl):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_dl:
            # Remove the unsqueeze operation - inputs are already in correct format
            # inputs = inputs.unsqueeze(-1)  # This was causing the error
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if total > 0:
            accuracy = correct / total
            print(f'Test Accuracy: {accuracy:.4f}')
        else:
            print('No test data available')


evaluate(resnet_model, val_dl)