# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import os, time
import numpy as np
import random
random.seed(42)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

import torch
torch.manual_seed(42)
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.dataset import Dataset
from torchvision.models import resnet
from torchvision import transforms, datasets, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
def load_transform_images(images_path, presplit, train_split, test_split, val_split, batch_size, threads, mean, std):
    train_transform = transforms.Compose([
                                         #transforms.RandomRotation(degrees=15),
                                         #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                         #transforms.RandomResizedCrop((224,224)),
                                         transforms.Resize((224,224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(torch.Tensor(mean),
                                                              torch.Tensor(std))])

    test_transform = transforms.Compose([
                                        transforms.Resize((224,224)),
                                        #transforms.CenterCrop((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(torch.Tensor(mean),
                                                             torch.Tensor(std))])

    val_transform = transforms.Compose([
                                       transforms.Resize((224,224)),
                                       #transforms.CenterCrop((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(torch.Tensor(mean),
                                                            torch.Tensor(std))])
    if presplit:
        try:
            training_set = datasets.ImageFolder(root=images_path+'/train', transform=train_transform)
            validation_set = datasets.ImageFolder(root=images_path+'/val', transform=val_transform)
        except FileNotFoundError:
            raise Exception('Not presplit into Training and Validation sets')
        try:
            testing_set = datasets.ImageFolder(root=images_path+'/test', transform=test_transform)
        except:
            testing_set = validation_set
        dataset = training_set
    else:
        dataset = datasets.ImageFolder(root=images_path, transform=train_transform)
        train_size = int(train_split * len(dataset))
        test_size = int(test_split * len(dataset))
        val_size = len(dataset) - train_size - test_size
        training_set, testing_set, validation_set = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    training_set_loader = DataLoader(training_set, batch_size=batch_size, num_workers=threads, shuffle=True)
    validation_set_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=threads, shuffle=True)
    testing_set_loader = DataLoader(testing_set, batch_size=batch_size, num_workers=threads, shuffle=False)

    return training_set_loader, testing_set_loader, validation_set_loader, dataset, training_set, testing_set, validation_set

images_path = 'data_small/images/Images/'
results_path = images_path+'_results'
presplit = False
train_split = 0.5
val_split = 0.25
test_split = 0.25
batch_size = 128
threads = 0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

training_set_loader, testing_set_loader, validation_set_loader, dataset, training_set, testing_set, validation_set = \
                  load_transform_images(images_path, presplit, train_split, test_split, val_split, batch_size, threads, mean, std)



class_names = dataset.classes
class_names = [classes[10:] for classes in class_names]
classes = ('Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu', 'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black-and-tan_coonhound', 'Walker_hound', 'English_foxhound', 'redbone', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier', 'Norwich_terrier', 'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier', 'Sealyham_terrier', 'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 'soft-coated_wheaten_terrier', 'West_Highland_white_terrier', 'Lhasa', 'flat-coated_retriever', 'curly-coated_retriever', 'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 'German_short-haired_pointer', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel', 'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd', 'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog', 'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle', 'standard_poodle', 'Mexican_hairless', 'dingo', 'dhole', 'African_hunting_dog')
print(class_names)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# import torch
# import torch.nn as nn
# import torchvision
# import torch.nn.functional as F
# 
# # Define a convolution neural network
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(12)
#         self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(12)
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(24)
#         self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(24)
#         self.fc1 = nn.Linear(24*10*10, 10)
# 
#     def forward(self, input):
#         output = F.relu(self.bn1(self.conv1(input)))      
#         output = F.relu(self.bn2(self.conv2(output)))     
#         output = self.pool(output)                        
#         output = F.relu(self.bn4(self.conv4(output)))     
#         output = F.relu(self.bn5(self.conv5(output)))     
#         output = output.view(-1, 24*10*10)
#         output = self.fc1(output)
# 
#         return output
# 
# # Instantiate a neural network model 
# model = Network()

# === AFTER (edited) ===
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*107*107, 120)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24*107*107)
        output = self.fc1(output)

        return output


model = Network()

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
from torch.optim import Adam
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# from torch.autograd import Variable
# 
# # Function to save the model
# def saveModel():
#     path = "data_small/myFirstModel.pth"
#     torch.save(model.state_dict(), path)
# 
# # Function to test the model with the test dataset and print the accuracy for the test images
# def testAccuracy():
#     
#     model.eval()
#     accuracy = 0.0
#     total = 0.0
#     
#     with torch.no_grad():
#         for data in testing_set_loader: # for data in test_loader: # fix for crash isolation reasons
#             images, labels = data
#             # run the model on the test set to predict labels
#             outputs = model(images)
#             # the label with the highest energy will be our prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             accuracy += (predicted == labels).sum().item()
#     
#     # compute the accuracy over all test images
#     accuracy = (100 * accuracy / total)
#     return(accuracy)
# 
# 
# # Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
# def train(num_epochs):
#     
#     best_accuracy = 0.0
# 
#     # Define your execution device
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("The model will be running on", device, "device")
#     # Convert model parameters and buffers to CPU or Cuda
#     model.to(device)
# 
#     for epoch in range(num_epochs):  # loop over the dataset multiple times
#         running_loss = 0.0
#         running_acc = 0.0
# 
#         for i, (images, classes) in enumerate(training_set_loader, 0): # for i, (images, classes) in enumerate(dataset, 0): # fix for crash isolation reasons
#             
#             # get the inputs
#             images = Variable(images.to(device))
#             
#             classes = torch.tensor(classes)
#             classes = Variable(classes.to(device))
# 
#             # zero the parameter gradients
#             optimizer.zero_grad()
#             # predict classes using images from the training set
#             outputs = model(device)
#             # compute the loss based on model output and real labels
#             loss = loss_fn(outputs, class_names)
#             # backpropagate the loss
#             loss.backward()
#             # adjust parameters based on the calculated gradients
#             optimizer.step()
# 
#             # Let's print statistics for every 1,000 images
#             running_loss += loss.item()     # extract the loss value
#             if i % 1000 == 999:    
#                 # print every 1000 (twice per epoch) 
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 1000))
#                 # zero the loss
#                 running_loss = 0.0
# 
#         # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
#         accuracy = testAccuracy()
#         print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
#         
#         # we want to save the model if the accuracy is the best
#         if accuracy > best_accuracy:
#             saveModel()
#             best_accuracy = accuracy

# === AFTER (edited) ===
from torch.autograd import Variable


def saveModel():
    path = "data_small/myFirstModel.pth"
    torch.save(model.state_dict(), path)


def testAccuracy():

    model.eval()
    accuracy = 0.0
    total = 0.0
    
    device = next(model.parameters()).device

    with torch.no_grad():
        for data in testing_set_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()


    accuracy = (100 * accuracy / total)
    return(accuracy)



def train(num_epochs):

    best_accuracy = 0.0


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, classes) in enumerate(training_set_loader, 0):


            images = Variable(images.to(device))

            classes = torch.tensor(classes)
            classes = Variable(classes.to(device))


            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_fn(outputs, classes)

            loss.backward()

            optimizer.step()


            running_loss += loss.item()
            if i % 1000 == 999:

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))

                running_loss = 0.0


        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))


        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
import matplotlib.pyplot as plt
import numpy as np

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader  
    images, labels = next(iter(testing_set_loader)) # images, labels = next(iter(test_loader)) # fix for crash isolation reasons

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 7}
if __name__ == "__main__":
    
    # Let's build our model
    train(2) # 5
    print('Finished Training')

    # Test which classes performed well
    testAccuracy() # testModelAccuracy() # fix for crash isolation reasons
    
    # Let's load the model we just created and test the accuracy per label
    model = Network()
    path = "myFirstModel.pth"
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch()