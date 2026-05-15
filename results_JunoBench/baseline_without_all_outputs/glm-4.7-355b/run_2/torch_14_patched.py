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
import xml.etree.ElementTree as ET


def transform_target(target):
    """Convert VOC XML format to Faster R-CNN expected format"""
    # Parse annotation if it's a dictionary with 'annotation' key
    if isinstance(target, dict) and 'annotation' in target:
        annotation = target['annotation']
    else:
        annotation = target
    
    # Extract objects
    objects = annotation.get('object', [])
    if not isinstance(objects, list):
        objects = [objects]
    
    boxes = []
    labels = []
    
    for obj in objects:
        # Get bounding box
        bndbox = obj['bndbox']
        xmin = float(bndbox['xmin'])
        ymin = float(bndbox['ymin'])
        xmax = float(bndbox['xmax'])
        ymax = float(bndbox['ymax'])
        boxes.append([xmin, ymin, xmax, ymax])
        
        # Get class name and convert to label
        # VOC has 20 classes + 1 for background
        class_names = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        class_name = obj['name']
        label = class_names.index(class_name) + 1  # +1 for background class
        labels.append(label)
    
    # Convert to tensors
    target_dict = {}
    if len(boxes) > 0:
        target_dict['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
    else:
        target_dict['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
    
    target_dict['labels'] = torch.as_tensor(labels, dtype=torch.int64)
    
    return target_dict


class VOCDetectionWrapper(torch.utils.data.Dataset):
    """Wrapper to convert VOC dataset to the format needed by Faster R-CNN"""
    def __init__(self, voc_dataset):
        self.voc_dataset = voc_dataset
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __getitem__(self, idx):
        img, target = self.voc_dataset[idx]
        # Convert target to Faster R-CNN format
        target = transform_target(target)
        return img, target
    
    def __len__(self):
        return len(self.voc_dataset)


def collate_fn(batch):
    """Custom collate function for object detection"""
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return images, targets


# Load the VOC datasets without transform (they return PIL images)
trainset_base = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='train', download=False)
testset_base = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='val', download=False)

# Wrap the datasets
trainset = VOCDetectionWrapper(trainset_base)
testset = VOCDetectionWrapper(testset_base)

# Create dataloaders with custom collate function
trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

# Load model with updated pretrained parameter
model = fasterrcnn_resnet50_fpn(weights=None)

# Modify the model for our number of classes (20 + 1 for background)
num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Set model to training mode
model.train()

# Create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(trainloader, 0):
        # Move data to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        # Forward pass with targets (returns losses during training)
        loss_dict = model(images, targets)
        
        # Sum all losses
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        # Backward pass
        losses.backward()
        optimizer.step()

        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss_value))

print('Finished Training')