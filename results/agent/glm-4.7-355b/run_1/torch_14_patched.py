# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
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


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# VOC class names (including background as class 0)
VOC_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def collate_fn(batch):
    return tuple(zip(*batch))

def convert_voc_annotation_to_target(annotation):
    """Convert VOCDetection annotation to Faster R-CNN target format."""
    objects = annotation['annotation']['object']
    
    boxes = []
    labels = []
    areas = []
    iscrowd = []
    
    for obj in objects:
        bbox = obj['bndbox']
        xmin = float(bbox['xmin'])
        ymin = float(bbox['ymin'])
        xmax = float(bbox['xmax'])
        ymax = float(bbox['ymax'])
        boxes.append([xmin, ymin, xmax, ymax])
        
        class_name = obj['name']
        label_idx = VOC_CLASSES.index(class_name) if class_name in VOC_CLASSES else 0
        labels.append(label_idx)
        
        area = (xmax - xmin) * (ymax - ymin)
        areas.append(area)
        
        iscrowd.append(int(obj['difficult']))
    
    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64),
        'image_id': torch.tensor([int(annotation['annotation']['filename'].split('.')[0].split('_')[-1])]),
        'area': torch.tensor(areas, dtype=torch.float32),
        'iscrowd': torch.tensor(iscrowd, dtype=torch.int64)
    }


trainset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='train', download=False, transform=transform)
testset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='val', download=False, transform=transform)


trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)


model = fasterrcnn_resnet50_fpn(pretrained=True)


num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


num_epochs = 1
for epoch in range(num_epochs):
    for i, (images, annotations) in enumerate(trainloader, 0):
        targets = [convert_voc_annotation_to_target(ann) for ann in annotations]
        
        images = list(images)
        model.train()
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, losses.item()))

print('Finished Training')