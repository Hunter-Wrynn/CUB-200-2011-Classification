import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import alpha_clip
from bird2 import Bird

# Step 1: Load class names
def load_class_texts(classes_txt_path):
    with open(classes_txt_path, 'r') as f:
        class_texts = [
            ' '.join(line.strip().split(' ', 1)[1].split('.')[1].replace('_', ' ').split())
            for line in f.readlines()
        ]
    return class_texts

# Step 2: Define Dataset Class
class BirdDataset(Dataset):
    def __init__(self, bird_obj, preprocess, mask_transform):
        self.bird = bird_obj
        self.preprocess = preprocess
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.bird.data)

    def __getitem__(self, idx):
        image, mask, class_name = self.bird.get_item_by_index(idx)
        image_tensor = self.preprocess(image)
        binary_mask = mask.astype(np.uint8)
        if len(binary_mask.shape) == 2:
            binary_mask = binary_mask
        elif len(binary_mask.shape) == 3:
            binary_mask = binary_mask[:, :, 0]
        alpha = self.mask_transform((binary_mask * 255).astype(np.uint8))
        return image_tensor, alpha, class_name

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
model_path = "/ailab/user/mahaoxuan/BIRD/birdDataSet_mixed/models/alpha_clip_last.pth"
classes_txt_path = "/ailab/user/mahaoxuan/BIRD/birdDataSet_mixed/bird/classes.txt"
images_dir = "/ailab/user/mahaoxuan/BIRD/birdDataSet_mixed/bird/images"
train_txt_path = "/ailab/user/mahaoxuan/BIRD/birdDataSet_mixed/bird/trainDataSet_mixed.txt"
sam_path="/ailab/user/mahaoxuan/BIRD/sam_vit_b_01ec64.pth"
# Load the model without weights initially
model, preprocess = alpha_clip.load(
    "ViT-L/14",
    alpha_vision_ckpt_pth="None",  # Pass the string "None" here
    device=device
)
model.eval()

# Load and process the state dict
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

state_dict = torch.load(model_path, map_location=device)
state_dict = remove_module_prefix(state_dict)
model.load_state_dict(state_dict)

# Load class texts
class_texts = load_class_texts(classes_txt_path)

# Mask transformation
mask_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(0.5, 0.26)
])

# Initialize Bird object and dataset
bird = Bird(images_dir, train_txt_path, classes_txt_path, sam_path)
dataset = BirdDataset(bird, preprocess, mask_transform)

# Step 4: Sample 100 data points and create DataLoader
subset_indices = np.random.choice(len(dataset), 100, replace=False)
subset_loader = DataLoader(dataset, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(subset_indices))

# Tokenizer
tokenizer = alpha_clip.tokenize

# Tokenize class texts
class_tokens = tokenizer(class_texts).to(device)
with torch.no_grad():
    class_features = model.encode_text(class_tokens)  # shape: [70, feature_dim]

# Step 5: Calculate prediction accuracy
correct_predictions = 0

for image_tensor, alpha, true_class in tqdm(subset_loader, desc="Evaluating"):
    image_tensor = image_tensor.to(device).to(torch.float16)  # 转换为 float16
    alpha = alpha.to(device).to(torch.float16)                # alpha 也转换为 float16
    true_class = true_class[0]  # 因为 batch_size=1，所以取第一个

    with torch.no_grad():
        image_feature = model.visual(image_tensor, alpha)  # shape: [1, feature_dim]
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)  # L2 归一化


        # Compute similarities
        similarities = (image_feature @ class_features.T).squeeze(0)  # shape: [70]
        predicted_class_index = similarities.argmax().item()
        predicted_class_name = class_texts[predicted_class_index]

        # Compare predicted result with true class
        if predicted_class_name == true_class:
            correct_predictions += 1

accuracy = correct_predictions / len(subset_loader)
print(f"Accuracy on 100 samples: {accuracy * 100:.2f}%")
