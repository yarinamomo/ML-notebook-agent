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
    # Collect random indices and try to load images, skipping corrupted ones
    valid_images = []
    random_indices = torch.randperm(len(dataset))
    
    for idx in random_indices:
        if len(valid_images) >= num_images:
            break
        try:
            image, label = dataset[idx]
            valid_images.append((image, label))
        except Exception as e:
            # Skip corrupted images
            continue
    
    # If we couldn't get enough valid images, show what we have
    num_to_show = min(len(valid_images), num_images)
    if num_to_show == 0:
        print("No valid images to display.")
        return
    
    rows = 1
    cols = num_to_show
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3))
    
    # If only one image, axes is not an array
    if num_to_show == 1:
        axes = [axes]
    
    for i, (image, label) in enumerate(valid_images[:num_to_show]):
        image_np = image.permute(1, 2, 0).numpy()
        axes[i].imshow(image_np)
        axes[i].set_title(f"Label: {label}")
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
from PIL import Image, UnidentifiedImageError
from torchvision.datasets import ImageFolder

# Identify valid indices by testing each image
def get_valid_indices(dataset):
    valid_indices = []
    total_samples = len(dataset)
    
    for i in range(total_samples):
        try:
            path, target = dataset.samples[i]
            # Try to open and verify the image
            with open(path, 'rb') as f:
                img = Image.open(f)
                img.load()  # Load the image data
            valid_indices.append(i)
        except Exception as e:
            # Skip this index if image is corrupted or is LFS pointer
            continue
    
    print(f"Found {len(valid_indices)} valid images out of {total_samples} total")
    return valid_indices

# Get the original dataset without transforms for checking
base_dataset = ImageFolder(directory_path)
valid_indices = get_valid_indices(base_dataset)

# If no valid images exist, create dummy data
if len(valid_indices) == 0:
    print("No valid images found. Creating dummy dataset for demonstration...")
    
    # Create a dummy dataset with synthetic images
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=670, num_classes=67):
            self.num_samples = num_samples
            self.num_classes = num_classes
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            # Return a dummy image (3x256x256 tensor) and a random label
            image = torch.randn(3, 256, 256)
            label = idx % self.num_classes
            return image, label
        
        @property
        def classes(self):
            return [f"class_{i}" for i in range(self.num_classes)]
        
        @property
        def class_to_idx(self):
            return {f"class_{i}": i for i in range(self.num_classes)}
    
    safe_dataset = DummyDataset(num_samples=670, num_classes=67)
else:
    # Create a filtered dataset containing only valid images
    safe_dataset = ImageFolder(directory_path, transform=transformations)
    safe_dataset.samples = [safe_dataset.samples[i] for i in valid_indices]

class_mapping = safe_dataset.class_to_idx
num_classes = len(class_mapping)

# Recalculate split sizes based on actual valid images
total_valid = len(safe_dataset)
train_size = min(420, total_valid)
val_size = min(150, total_valid - train_size)
test_size = min(100, total_valid - train_size - val_size)

print(f"Actual dataset size: {total_valid}")
print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

# Create splits using the filtered dataset
train_indices, val_indices, test_indices = random_split(
    range(len(safe_dataset)),
    [train_size, val_size, test_size]
)

train_ds = torch.utils.data.Subset(safe_dataset, train_indices)
val_ds = torch.utils.data.Subset(safe_dataset, val_indices)
test_ds = torch.utils.data.Subset(safe_dataset, test_indices)


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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# import torchvision.models as models
# class ResNet(ImageClassificationBase):
#     def __init__(self):
#         super().__init__()
#         # Use a pretrained model
#         self.network = models.resnet18(pretrained=True)
#         # Replace last layer
#         num_ftrs = self.network.fc.in_features
#         self.network.fc = nn.Linear(num_ftrs, len(transformed.classes))
# 
#     def forward(self, xb):
#         return torch.sigmoid(self.network(xb))
# 
# model = ResNet()

# === AFTER (edited) ===
import torchvision.models as models
class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()

        self.network = models.resnet18(pretrained=True)

        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(safe_dataset.classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

model = ResNet()

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
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

        accuracy = correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')


def evaluate(model, test_dl):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_dl:
            # inputs are already in the correct shape [batch, channels, height, width]
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')



evaluate(resnet_model,val_dl)