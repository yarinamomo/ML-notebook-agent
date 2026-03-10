# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
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
    """Custom collate function for object detection with variable-sized images."""
    return tuple(zip(*batch))


def transform_target(target):
    """Convert VOC detection target to format expected by Faster R-CNN."""
    objects = target['annotation']['object']
    if isinstance(objects, dict):
        objects = [objects]
    
    boxes = []
    labels = []
    
    for obj in objects:
        xmin = float(obj['bndbox']['xmin'])
        ymin = float(obj['bndbox']['ymin'])
        xmax = float(obj['bndbox']['xmax'])
        ymax = float(obj['bndbox']['ymax'])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1)  # Use 1 for all objects for simplicity (foreground)
    
    if len(boxes) == 0:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
    else:
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
    
    target_dict = {}
    target_dict['boxes'] = boxes
    target_dict['labels'] = labels
    target_dict['image_id'] = torch.tensor([int(target['annotation']['filename'].split('.')[0])])
    target_dict['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    target_dict['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
    
    return target_dict


class VOCDetectionDataset(torchvision.datasets.VOCDetection):
    """Custom VOCDetection dataset with proper target formatting."""
    
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None):
        super().__init__(root=root, year=year, image_set=image_set, download=download)
        self.transform = transform
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        
        if self.transform is not None:
            img = self.transform(img)
        
        # Convert target format
        target = transform_target(target)
        
        return img, target


transform = transforms.Compose([
    transforms.ToTensor()
])


trainset = VOCDetectionDataset(root="data_small", year='2012', image_set='train', download=False, transform=transform)
testset = VOCDetectionDataset(root="data_small", year='2012', image_set='val', download=False, transform=transform)


trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)


model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1')


num_classes = 21  # VOC has 20 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


num_epochs = 1
model.train()
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(trainloader, 0):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        losses.backward()
        optimizer.step()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, losses.item()))

print('Finished Training')