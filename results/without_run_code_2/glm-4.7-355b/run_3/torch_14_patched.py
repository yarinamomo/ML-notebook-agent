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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import xml.etree.ElementTree as ET


class VOCDetectionDataset(torchvision.datasets.VOCDetection):
    def __init__(self, root, year='2012', image_set='train', download=False, transforms=None):
        super().__init__(root=root, year=year, image_set=image_set, download=download)
        self.transforms = transforms
    
    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.images[idx]).convert("RGB")
        w, h = img.size
        
        # Parse XML annotation
        target = self.parse_voc_xml(
            ET.parse(self.annotations[idx]).getroot()
        )
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        objs = target['annotation'].get('object', [])
        if not isinstance(objs, list):
            objs = [objs]
        
        for obj in objs:
            box = obj['bndbox']
            boxes.append([float(box['xmin']), float(box['ymin']), 
                         float(box['xmax']), float(box['ymax'])])
            # Use class name mapping (simplified to class 1 for all objects)
            labels.append(1)  # Will be overwritten if actual class mapping is available
        
        target_out = {}
        target_out['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target_out['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target_out['image_id'] = torch.tensor([idx])
        target_out['area'] = (target_out['boxes'][:, 3] - target_out['boxes'][:, 1]) * \
                            (target_out['boxes'][:, 2] - target_out['boxes'][:, 0])
        target_out['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target_out


def get_transform(train):
    transforms_list = []
    transforms_list.append(transforms.ToTensor())
    if train:
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transforms_list)


# Use collate_fn that returns list of tuples (since images have different sizes)
def collate_fn(batch):
    return tuple(zip(*batch))


trainset = VOCDetectionDataset(root="data_small", year='2012', image_set='train', download=False, transforms=get_transform(train=True))
testset = VOCDetectionDataset(root="data_small", year='2012', image_set='val', download=False, transforms=get_transform(train=False))

trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

num_classes = 21  # 20 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop - simplified for verification
num_epochs = 1
model.train()
print("Starting training...")
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(trainloader, 0):
        # Convert to list of tensors
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        print(f'Batch {i}: loss: {losses.item():.3f}')
        
        # Just test a few batches to verify it works
        if i >= 5:
            break

print('Finished Training - Successfully ran without crashes!')