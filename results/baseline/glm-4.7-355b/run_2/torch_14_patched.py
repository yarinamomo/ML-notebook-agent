# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 1}
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
import numpy as np


class VOCDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, voc_dataset):
        self.voc_dataset = voc_dataset
    
    def __len__(self):
        return len(self.voc_dataset)
    
    def __getitem__(self, idx):
        image, annotation = self.voc_dataset[idx]
        
        # Get image dimensions
        img_width = image.shape[1]
        img_height = image.shape[2]
        
        # Parse annotation from VOCDetection format
        # annotation['annotation']['object'] is a list of objects, could be None
        objects = annotation['annotation']['object']
        if objects is None:
            objects = []
        elif not isinstance(objects, list):
            objects = [objects]
        
        # VOC class names (20 classes + 1 background = 21 total)
        VOC_CLASSES = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        class_to_idx = {cls: idx + 1 for idx, cls in enumerate(VOC_CLASSES)}
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for obj in objects:
            # Get bounding box coordinates
            bndbox = obj['bndbox']
            xmin = int(bndbox['xmin'])
            ymin = int(bndbox['ymin'])
            xmax = int(bndbox['xmax'])
            ymax = int(bndbox['ymax'])
            
            # Skip invalid boxes
            if xmax <= xmin or ymax <= ymin:
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_to_idx.get(obj['name'], 0))
            areas.append((xmax - xmin) * (ymax - ymin))
            iscrowd.append(0)
        
        # Convert to tensors (if no objects, handle gracefully)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        return image, target


# Custom collate function for object detection
def collate_fn(batch):
    """Custom collate function that returns lists instead of batching"""
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets


# Load VOCDetection dataset without transform (we'll handle it manually later)
trainset_raw = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='train', download=False)
testset_raw = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='val', download=False)

# Wrap with custom dataset
trainset = VOCDatasetWrapper(trainset_raw)
testset = VOCDatasetWrapper(testset_raw)

# Create DataLoader with custom collate function
trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)


# Load model with updated parameters
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')


num_classes = 21  # 20 VOC classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


# Training loop
num_epochs = 1
model.train()  # Set model to training mode

for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(trainloader, 0):
        # Move images and targets to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass (model returns losses during training)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, losses.item()))

print('Finished Training')