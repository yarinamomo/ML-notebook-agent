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
from torch.utils.data import DataLoader, Subset


transform = transforms.Compose([
    transforms.ToTensor(),
])

# VOC class names (20 classes + background)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Use download=True to get the dataset if not present
trainset_full = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='train', download=True, transform=transform)
testset_full = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='val', download=True, transform=transform)

# Use a small subset for faster execution
trainset = Subset(trainset_full, range(20))  # Only 20 training images
testset = Subset(testset_full, range(10))    # Only 10 test images


def collate_fn(batch):
    """Custom collate function to handle variable-sized annotations"""
    return tuple(zip(*batch))

trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)


model = fasterrcnn_resnet50_fpn(weights='DEFAULT')


num_classes = 21  # VOC has 20 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


num_epochs = 1
model.train()
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(trainloader, 0):
        images = [image.to(device) for image in images]
        
        # Convert VOCDetection targets to the format expected by Faster R-CNN
        formatted_targets = []
        for target in targets:
            # VOC format: {'annotation': {'object': [{'name': class_name, 'bndbox': {...}}, ...]}}
            boxes = []
            labels = []
            for obj in target['annotation']['object']:
                bbox = obj['bndbox']
                boxes.append([float(bbox['xmin']), float(bbox['ymin']), 
                            float(bbox['xmax']), float(bbox['ymax'])])
                # Convert class name to index (VOC classes are numbered 1-20)
                labels.append(VOC_CLASSES.index(obj['name']) + 1)
            
            if len(boxes) == 0:
                # Handle images with no objects
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
            
            formatted_targets.append({
                'boxes': boxes,
                'labels': labels
            })
        
        optimizer.zero_grad()
        
        # Faster R-CNN computes loss automatically during training
        loss_dict = model(images, formatted_targets)
        
        # Sum all losses
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, losses.item()))

print('Finished Training')