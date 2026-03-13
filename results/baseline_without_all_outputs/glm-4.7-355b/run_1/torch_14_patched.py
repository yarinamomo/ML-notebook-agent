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
import torch.nn.functional as F


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load VOC Detection dataset
trainset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='train', download=False, transform=transform)
testset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='val', download=False, transform=transform)


# Helper to convert VOC annotations to Faster R-CNN format
def convert_voc_annotation(annot, num_classes=21):
    objects = annot['annotation']['object']
    if isinstance(objects, dict):
        objects = [objects]
    
    boxes = []
    labels = []
    
    # Class names in VOC dataset
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    class_to_idx = {cls: i+1 for i, cls in enumerate(voc_classes)}  # +1 because 0 is background
    
    for obj in objects:
        class_name = obj['name']
        if class_name in class_to_idx:
            labels.append(class_to_idx[class_name])
        else:
            labels.append(0)  # background if unknown
            
        bbox = obj['bndbox']
        boxes.append([
            float(bbox['xmin']), 
            float(bbox['ymin']), 
            float(bbox['xmax']), 
            float(bbox['ymax'])
        ])
    
    num_objs = len(boxes)
    
    target = {}
    target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
    target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
    target["image_id"] = torch.tensor([0])
    target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
    target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
    
    return target


# Custom collate function for DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))


trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)


model = fasterrcnn_resnet50_fpn(pretrained=True)


num_classes = 21  # 20 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


num_epochs = 1
model.train()  # Set model to training mode
for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        images, annotations = data
        
        # Convert VOC annotations to Faster R-CNN format
        targets = [convert_voc_annotation(ann) for ann in annotations]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with both images and targets for training
        loss_dict = model(images, targets)
        
        # Sum all losses
        loss = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        loss.backward()
        optimizer.step()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

print('Finished Training')