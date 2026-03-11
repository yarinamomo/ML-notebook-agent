# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
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


class VDTDetectionTransform:
    """Convert VOC detection targets to format needed by Faster R-CNN"""
    def __init__(self):
        # VOC dataset has 20 classes (0-19), we use 0 for background
        self.voc_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.voc_classes)}
    
    def __call__(self, target):
        boxes = []
        labels = []
        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            boxes.append([float(bbox['xmin']), float(bbox['ymin']), 
                         float(bbox['xmax']), float(bbox['ymax'])])
            # Convert string label to integer class ID
            label_name = obj['name']
            labels.append(self.class_to_idx.get(label_name, 0))
        
        return {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = VDTDetectionTransform()


trainset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='train', download=False, transform=transform, target_transform=target_transform)
testset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='val', download=False, transform=transform, target_transform=target_transform)


def collate_fn(batch):
    """Custom collate function for object detection tasks"""
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append({
            'boxes': target['boxes'],
            'labels': target['labels']
        })
    return images, targets


trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)


model = fasterrcnn_resnet50_fpn(pretrained=True)


num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


num_epochs = 1
max_iterations = 5
for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs, labels)
        loss = sum(loss for loss in outputs.values())
        loss.backward()
        optimizer.step()

        if i % 5 == 4:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
        
        if i >= max_iterations:
            break

print('Finished Training')