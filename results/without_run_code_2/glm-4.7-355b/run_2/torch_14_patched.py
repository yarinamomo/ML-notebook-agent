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


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


trainset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='train', download=False, transform=transform)
testset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='val', download=False, transform=transform)


trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)


model = fasterrcnn_resnet50_fpn(weights='DEFAULT')


num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


# VOC class names
voc_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

model.train()
num_epochs = 1
max_batches = 10  # Limit to 10 batches for demonstration
for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        if i >= max_batches:
            break
            
        inputs, labels = data
        # Unwrap since DataLoader wraps everything in a list
        inputs = inputs[0]
        annotation = labels['annotation']
        
        # Extract boxes and labels from VOC format
        objects = annotation['object']
        if isinstance(objects, dict):
            objects = [objects]
        
        boxes = []
        class_labels = []
        for obj in objects:
            bbox = obj['bndbox']
            # bbox values are already in format ['value']
            xmin = float(bbox['xmin'][0]) if isinstance(bbox['xmin'], list) else float(bbox['xmin'])
            ymin = float(bbox['ymin'][0]) if isinstance(bbox['ymin'], list) else float(bbox['ymin'])
            xmax = float(bbox['xmax'][0]) if isinstance(bbox['xmax'], list) else float(bbox['xmax'])
            ymax = float(bbox['ymax'][0]) if isinstance(bbox['ymax'], list) else float(bbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            
            # Get class label from name
            name = obj['name'][0] if isinstance(obj['name'], list) else obj['name']
            class_idx = voc_classes.index(name) + 1 if name in voc_classes else 0
            class_labels.append(class_idx)
        
        if len(boxes) == 0:
            # Skip images with no objects
            continue
            
        target = {}
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(class_labels, dtype=torch.int64)
        
        optimizer.zero_grad()
        loss_dict = model([inputs], [target])
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, losses.item()))

print('Finished Training')