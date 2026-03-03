# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
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


# VOC class names to IDs (1-20 are the classes, 0 is reserved for background)
VOC_CLASSES = [
    '__background__',  # 0 (background)
    'aeroplane',       # 1
    'bicycle',         # 2
    'bird',            # 3
    'boat',            # 4
    'bottle',          # 5
    'bus',             # 6
    'car',             # 7
    'cat',             # 8
    'chair',           # 9
    'cow',             # 10
    'diningtable',     # 11
    'dog',             # 12
    'horse',           # 13
    'motorbike',       # 14
    'person',          # 15
    'pottedplant',     # 16
    'sheep',           # 17
    'sofa',            # 18
    'train',           # 19
    'tvmonitor'        # 20
]

def convert_voc_to_faster_rcnn_target(target, device):
    """Convert VOC annotation format to Faster R-CNN expected format."""
    annotation = target['annotation']
    objects = annotation['object']
    
    # Extract boxes and labels
    boxes = []
    labels = []
    
    for obj in objects:
        # Get bounding box coordinates (convert from string to int)
        bndbox = obj['bndbox']
        xmin = float(bndbox['xmin'])
        ymin = float(bndbox['ymin'])
        xmax = float(bndbox['xmax'])
        ymax = float(bndbox['ymax'])
        boxes.append([xmin, ymin, xmax, ymax])
        
        # Get class label
        class_name = obj['name']
        class_id = VOC_CLASSES.index(class_name)
        labels.append(class_id)
    
    # Convert to tensors
    boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.int64).to(device)
    
    # Create target dict in Faster R-CNN format
    converted = {
        'boxes': boxes,
        'labels': labels,
    }
    
    return converted


def collate_fn(batch):
    """Custom collate function that returns lists of images and targets."""
    images = []
    targets = []
    for item in batch:
        images.append(item[0])
        targets.append(item[1])
    return images, targets


transform = transforms.Compose([
    transforms.ToTensor(),
])


trainset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='train', download=True, transform=transform)
testset = torchvision.datasets.VOCDetection(root="data_small", year='2012', image_set='val', download=True, transform=transform)


# Use batch_size=1 and custom collate_fn for object detection
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)


model = fasterrcnn_resnet50_fpn(pretrained=True)


num_classes = 21  # 20 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


# Move model to device (CPU for this example)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


# Set model to training mode
model.train()


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


# Train on only a small subset (10 batches) to verify the code works
num_epochs = 1
max_batches = 10
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(trainloader):
        if i >= max_batches:
            break
        
        # Convert to list and move to device
        images = list(image.to(device) for image in images)
        targets = [convert_voc_to_faster_rcnn_target(t, device) for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, losses.item()))

print('Finished Training')