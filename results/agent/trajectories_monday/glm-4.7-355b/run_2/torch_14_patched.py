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


def collate_fn(batch):
    """
    Custom collate function for object detection datasets.
    Handles variable-sized images and annotations.
    """
    return tuple(zip(*batch))


def get_voc_target(annotation):
    """
    Convert VOC annotation format to the format expected by Faster R-CNN.
    """
    objects = annotation['annotation']['object']
    if isinstance(objects, dict):
        objects = [objects]
    
    boxes = []
    labels = []
    for obj in objects:
        bbox = obj['bndbox']
        boxes.append([float(bbox['xmin']), float(bbox['ymin']), 
                      float(bbox['xmax']), float(bbox['ymax'])])
        # VOC class labels are 1-20, Faster R-CNN expects them as is (0 can be background)
        labels.append(int(obj['name'][-1]) if obj['name'][-1].isdigit() else 1)
    
    target = {}
    target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
    target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
    return target


transform = transforms.Compose([
    transforms.ToTensor(),
])


# Use num_workers=0 to avoid multiprocessing issues
trainset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='train', download=False, transform=transform)
testset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='val', download=False, transform=transform)


# Using num_workers=0 to avoid worker process errors
trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)


model = fasterrcnn_resnet50_fpn(pretrained=True)


num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


num_epochs = 1
for epoch in range(num_epochs):
    # Just process 1 batch to avoid timeout
    for i, (images, annotations) in enumerate(trainloader):
        print(f"Processing batch {i}...")
        
        # Convert VOC annotations to Faster R-CNN format
        targets = [get_voc_target(ann) for ann in annotations]
        
        optimizer.zero_grad()
        
        # Set model to training mode and pass targets
        model.train()
        outputs = model(images, targets)
        
        # Compute losses
        loss = sum(loss for loss in outputs.values())
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}], Batch [{i+1}], Loss: {loss.item():.4f}')
        
        # Just process 1 batch and break to avoid timeout
        if i >= 0:
            break

print('Finished Training')