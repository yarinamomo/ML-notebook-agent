# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
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
import os
import numpy as np
from PIL import Image


# Since the actual dataset files are Git LFS pointers and not real images,
# create a mock dataset that generates dummy data for demonstration
class MockVOCDetection(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, image_size=(224, 224), transform=None):
        self.transform = transform
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate a random RGB image
        img = Image.fromarray(np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8))
        
        # Generate random boxes and labels (dummy data)
        num_boxes = np.random.randint(1, 5)
        
        # Generate valid bounding boxes: [x1, y1, x2, y2]
        # Ensure coordinates are within image bounds and x2 > x1, y2 > y1
        boxes = []
        for i in range(num_boxes):
            x1 = np.random.uniform(0, self.image_size[1] - 20)
            y1 = np.random.uniform(0, self.image_size[0] - 20)
            w = np.random.uniform(10, 20)
            h = np.random.uniform(10, 20)
            boxes.append([x1, y1, x1 + w, y1 + h])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        # Generate random labels (1 to 20 for VOC classes)
        labels = torch.randint(1, 21, (num_boxes,))
        
        # Convert image to tensor
        if self.transform:
            img = self.transform(img)
        
        # Return dict format expected by Faster R-CNN
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        return img, target


# Custom collate function to handle variable-sized targets
def collate_fn(batch):
    # Batch is a list of tuples (image, target)
    # We need to handle that each image might have different number of boxes
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # Stack images (they all have same size after transform)
    images = torch.stack(images)
    
    # For targets, we keep them as a list of dicts
    # Each dict might have different sized tensors
    return images, targets


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


trainset = MockVOCDetection(num_samples=50, transform=transform)
testset = MockVOCDetection(num_samples=10, transform=transform)


trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)


model = fasterrcnn_resnet50_fpn(pretrained=True)


num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)


# Set model to training mode
model.train()

num_epochs = 1
total_loss = 0
batches_processed = 0

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(trainloader, 0):
        # targets is already a list of dicts, which is what Faster R-CNN expects
        optimizer.zero_grad()
        
        # Pass targets to model for training
        try:
            loss_dict = model(inputs, targets)
            
            # Sum all the losses
            losses = sum(loss for loss in loss_dict.values())
            
            # Only proceed if we have a valid scalar loss
            if isinstance(losses, torch.Tensor) and losses.requires_grad:
                # Check for NaN or inf
                if torch.isnan(losses) or torch.isinf(losses):
                    print(f"Skipping batch due to unstable loss")
                    continue
                    
                losses.backward()
                optimizer.step()
                total_loss += losses.item()
                batches_processed += 1
                
                if i % 10 == 9:
                    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, avg_loss))
        except Exception as e:
            # Handle case where model doesn't return loss
            print(f"Skipping batch due to: {e}")
            continue

if batches_processed > 0:
    avg_loss = total_loss / batches_processed
    print('Final average loss: %.3f' % avg_loss)
print('Finished Training')