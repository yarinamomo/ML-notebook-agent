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
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


# Reverse the order: normalize PIL image, then convert to tensor
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ToTensor(),
])


trainset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='train', download=False, transform=transform)
testset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='val', download=False, transform=transform)


# Custom collate function for object detection (handles variable-sized images)
def collate_fn(batch):
    """Collate function that handles variable-sized images and returns format expected by Faster R-CNN."""
    images = []
    targets = []
    
    for sample in batch:
        if isinstance(sample, (list, tuple)) and len(sample) == 2:
            img, annotation = sample
            images.append(img)
            
            # Parse VOC annotation format
            if 'annotation' in annotation:
                anno = annotation['annotation']
                boxes = []
                labels = []
                
                if 'object' in anno:
                    objects = anno['object']
                    if not isinstance(objects, list):
                        objects = [objects]
                    
                    for obj in objects:
                        bndbox = obj['bndbox']
                        xmin = float(bndbox['xmin'])
                        ymin = float(bndbox['ymin'])
                        xmax = float(bndbox['xmax'])
                        ymax = float(bndbox['ymax'])
                        boxes.append([xmin, ymin, xmax, ymax])
                        
                        # VOC has 20 classes; using 1-based labels for 20 classes
                        class_idx = int(obj['name']) if obj['name'].isdigit() else 1
                        labels.append(class_idx)
                
                target = {}
                if len(boxes) > 0:
                    target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
                    target['labels'] = torch.tensor(labels, dtype=torch.int64)
                else:
                    target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    target['labels'] = torch.zeros((0,), dtype=torch.int64)
                targets.append(target)
            else:
                targets.append({'boxes': torch.zeros((0, 4), dtype=torch.float32), 
                                'labels': torch.zeros((0,), dtype=torch.int64)})
    
    return images, targets


trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)


model = fasterrcnn_resnet50_fpn(pretrained=True)


num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


num_epochs = 1
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(trainloader, 0):
        # Move images to device
        images = [img.to(device) for img in images]
        
        # Move targets to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        # Pass targets to model for loss computation
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, losses.item()))

print('Finished Training')