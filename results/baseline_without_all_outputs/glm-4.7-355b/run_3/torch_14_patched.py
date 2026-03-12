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
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision.models.detection.faster_rcnn as faster_rcnn


class VOCDetectionTransform:
    def __call__(self, image, target):
        image = transforms.ToTensor()(image)
        target_formatted = {
            'boxes': torch.as_tensor(
                [[obj['bndbox']['xmin'], obj['bndbox']['ymin'], 
                  obj['bndbox']['xmax'], obj['bndbox']['ymax']] 
                 for obj in target['annotation']['object']], 
                dtype=torch.float32
            ),
            'labels': torch.as_tensor(
                [self._class_to_int(obj['name']) for obj in target['annotation']['object']], 
                dtype=torch.int64
            )
        }
        return image, target_formatted
    
    def _class_to_int(self, class_name):
        voc_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        try:
            return voc_classes.index(class_name) + 1
        except ValueError:
            return 0  # background


collate_fn = lambda batch: batch


trainset = torchvision.datasets.VOCDetection(
    root="data_small", year='2012', image_set='train', 
    download=False, transform=VOCDetectionTransform()
)
testset = torchvision.datasets.VOCDetection(
    root="data_small", year='2012', image_set='val', 
    download=False, transform=VOCDetectionTransform()
)


trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)


model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')


num_classes = 21
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)


optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


num_epochs = 1
model.train()
for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data[0]
        inputs = inputs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        optimizer.zero_grad()
        loss_dict = model(inputs, [targets])
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, losses.item()))

print('Finished Training')