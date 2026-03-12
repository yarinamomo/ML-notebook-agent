# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
from torchvision.io import read_image
from torchvision.models import vit_b_16, ViT_B_16_Weights, list_models
from torchvision.datasets import ImageNet, ImageFolder
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
import torch
import time
from torchvision.transforms import transforms

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
img = read_image("data_small/10/ILSVRC2012_val_00037698.jpeg")
print(img.shape[0])
if img.shape[0] == 1:
     img = img.expand(3, -1, -1)
print(img.size())

# Step 1: Initialize model with the best available weights
weights =ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
model = vit_b_16(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
print(class_id)
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# Move model and data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used is: " + str(device))

# create a metric for accuracy
metric = MulticlassAccuracy(num_classes=1000).to(device)

# Step 2: Initialize the inference transforms
preprocess = weights.transforms(antialias=True)

preprocess_w_gray2rgb = transforms.Compose([
    lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x,
    preprocess
])
imagenet_val_dir = 'data_small'
dataset = ImageFolder(root=imagenet_val_dir, loader=read_image, transform=preprocess_w_gray2rgb)
class_dict = dataset.class_to_idx
class_dict = {value: key for key, value in class_dict.items()}

# Create a DataLoader to load the images in batches
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
def check_label_name(predictions, weights):
    for prediction in predictions:
        class_id = prediction.argmax().item()
        print(class_id)
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]
        print(f"{category_name}: {100 * score:.1f}%")
        print("\n")
        
        
def model_quantization(model, backend='x86', save=False):
    # Use 'x86' for server inference (the old 'fbgemm' is still available but 'x86' is the recommended default) and ``qnnpack`` for mobile inference.
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend

    quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
    scripted_quantized_model = torch.jit.script(quantized_model)
    if save:
        scripted_quantized_model.save("vit_scripted_quantized.pt")
        
        
def labels_process(labels, class_dict):
    # Change labels because of dataset idx
    labels = [class_dict[int(label)] for label in labels]
    labels = [int(num) for num in labels]
    labels = torch.tensor(labels)
    return labels
    
def inference(model, dataloader, class_dict, device, image_num_stop=40000):
    index_stop = image_num_stop // 8
    total_correct = 0
    total_samples = 0
    start_time = time.time()
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for index, (images, labels) in enumerate(dataloader):
            images = images.to(device)
        
            # Change labels because of dataset idx
            labels = labels_process(labels, class_dict)
            labels = labels.to(device)
        
            predictions = model(images)
            predicted_labels = torch.argmax(predictions, dim=1) + 1  # Add 1 to predicted labels
            
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
        
            if index % 50 == 0:
                print("{} images were processed out of 50,000".format(8 * index))
            
            if index == index_stop:
                print("Number of images processed: {} stopping now".format(index_stop*8))
                print("stopped checking because of errors for the entire dataset \n ")
                break
            
    accuracy = total_correct / total_samples
    end_time = time.time()
    duration = (end_time - start_time) / 60
    
    return accuracy, duration

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# # Step 1: Initialize model with the best available weights
# weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
# model = vit_b_16(weights=weights)
# quantized_vit = model_quantization(model=model, save=True)
# 
# accuracy, duration = inference(model=model, dataloader=dataloader, class_dict=class_dict, device=device, image_num_stop=100)
# 
# print("Inference took {} minutes".format(duration))
# print("Accuracy for this model is {}".format(accuracy))

# === AFTER (edited) ===
weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
model = vit_b_16(weights=weights)

accuracy, duration = inference(model=model, dataloader=dataloader, class_dict=class_dict, device=device, image_num_stop=100)

print("Inference took {} minutes".format(duration))
print("Accuracy for this model is {}".format(accuracy))