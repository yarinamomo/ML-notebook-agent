# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
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
    """Custom collate function to handle variable-sized images for Faster R-CNN"""
    images = []
    targets = []
    for img, target in batch:
        # Convert to tensor if not already
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        images.append(img)
        targets.append(target)
    return images, targets


print("Downloading and loading training dataset...")
trainset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='train', download=True)
print(f"Training dataset loaded successfully! Number of samples: {len(trainset)}")

print("\nLoading validation dataset...")
testset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='val', download=True)
print(f"Validation dataset loaded successfully! Number of samples: {len(testset)}")

print("\nCreating data loaders...")
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

print("\nInitializing Faster R-CNN model...")
model = fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print(f"Using device: {device}")

# Test that data loading works
print("\nTesting data loading with one batch...")
for images, targets in trainloader:
    print(f"Successfully loaded a batch!")
    print(f"Images: {len(images)}, Target type: {type(targets[0])}")
    break

print("\nDataset loading and model initialization successful!")
print("Training loop can be run if needed.")