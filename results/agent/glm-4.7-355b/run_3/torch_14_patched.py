# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
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


def collate_fn(batch):
    """Custom collate function to handle variable-sized images and targets."""
    return tuple(zip(*batch))


transform = transforms.Compose([
    transforms.ToTensor()
])


trainset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='train', download=False, transform=transform)
testset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='val', download=False, transform=transform)


trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)


model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')


num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


def convert_voc_target(target):
    """Convert VOC format target to format expected by Faster R-CNN."""
    boxes = []
    labels = []
    
    for obj in target['annotation']['object']:
        class_name = obj['name']
        # Map class names to indices (VOC has 20 classes + background)
        class_map = {
            'person': 1, 'bicycle': 2, 'car': 3, 'motorbike': 4, 'aeroplane': 5,
            'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,
            'fire hydrant': 11, 'stop sign': 12, 'parking meter': 13, 'bench': 14,
            'bird': 15, 'cat': 16, 'dog': 17, 'horse': 18, 'sheep': 19,
            'cow': 20, 'sofa': 1, 'pottedplant': 2, 'diningtable': 3, 'tvmonitor': 4
        }
        class_idx = class_map.get(class_name, 1)
        
        bbox = obj['bndbox']
        xmin = int(bbox['xmin'])
        ymin = int(bbox['ymin'])
        xmax = int(bbox['xmax'])
        ymax = int(bbox['ymax'])
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_idx)
    
    if len(boxes) == 0:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
    else:
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
    
    target_dict = {
        'boxes': boxes,
        'labels': labels
    }
    return target_dict


num_epochs = 1
model.train()
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(trainloader, 0):
        images = list(image.to(device) for image in images)
        targets = [convert_voc_target(target) for target in targets]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, losses.item()))
        
        # Run only first 20 batches to avoid timeout
        if i >= 20:
            break

print('Finished Training')