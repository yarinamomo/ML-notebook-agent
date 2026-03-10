# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'timeout', 'done': True, 'execution_count': None}
# === BEFORE (original) ===
# import torch
# import torchvision
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.transforms import transforms
# from torch.utils.data import DataLoader
# 
# # 定义转换
# transform = transforms.Compose([
#     transforms.ToTensor(), 
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# 
# # 加载数据集
# trainset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='train', download=False, transform=transform)
# testset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='val', download=False, transform=transform)
# 
# # 创建数据加载器
# trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
# testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
# 
# # 加载预训练的模型
# model = fasterrcnn_resnet50_fpn(pretrained=True)
# 
# # 替换分类器
# num_classes = 21  # 20 类 + 背景
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
# 
# # 定义优化器
# optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
# 
# # 训练模型
# num_epochs = 1 #10
# for epoch in range(num_epochs):
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = sum(loss for loss in outputs.values())
#         loss.backward()
#         optimizer.step()
# 
#         if i % 2000 == 1999:    # 每 2000 mini-batches 打印一次
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
# 
# print('Finished Training')

# === AFTER (edited) ===
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os
import xml.etree.ElementTree as ET
import numpy as np


class VOCDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, year='2012', image_set='train', transform=None):
        self.voc = torchvision.datasets.VOCDetection(root=root, year=year, image_set=image_set, download=False)
        self.transform = transform
        self.classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                       'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.voc)
    
    def __getitem__(self, idx):
        img, target = self.voc[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        # Get annotation file path
        annotation = target['annotation']
        objects = annotation['object']
        
        if not isinstance(objects, list):
            objects = [objects]
        
        boxes = []
        labels = []
        
        for obj in objects:
            class_name = obj['name']
            if class_name in self.class_to_idx:
                labels.append(self.class_to_idx[class_name])
                bndbox = obj['bndbox']
                boxes.append([float(bndbox['xmin']), float(bndbox['ymin']),
                            float(bndbox['xmax']), float(bndbox['ymax'])])
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        
        return img, target


# Define transform for detection
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


# Create datasets
trainset = VOCDetectionDataset(root="data_small", year='2012', image_set='train', transform=get_transform())
testset = VOCDetectionDataset(root="data_small", year='2012', image_set='val', transform=get_transform())


# Custom collate function for detection
def collate_fn(batch):
    return tuple(zip(*batch))


# Create dataloaders with custom collate function
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)


# Load pretrained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)


# Replace the classifier head for custom number of classes
num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


# Setup optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


# Training loop
model.train()
num_epochs = 1
for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        inputs = list(image.to(torch.float32) for image in data[0])
        targets = [{k: v for k, v in t.items()} for t in data[1]]
        
        # Move targets to same device as inputs if needed
        for target in targets:
            for key in target:
                if isinstance(target[key], torch.Tensor):
                    target[key] = target[key].to(inputs[0].device)
        
        optimizer.zero_grad()
        
        # Forward pass with targets to get losses
        loss_dict = model(inputs, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        losses.backward()
        optimizer.step()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, losses.item()))

print('Finished Training')