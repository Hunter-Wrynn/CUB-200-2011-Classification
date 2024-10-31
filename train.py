# train_alpha_clip_distributed.py

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
import numpy as np
import alpha_clip
from segment_anything import sam_model_registry, SamPredictor
from bird2 import Bird
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# 自定义 BirdDataset 和 clip_loss

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

def clip_loss(image_features, text_features, temperature=0.1):
    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
    logits_per_image = image_features @ text_features.t() / temperature
    logits_per_text = text_features @ image_features.t() / temperature
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size).to(image_features.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2

def train(rank, args):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=args.world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # 加载模型和预处理
    model, preprocess = alpha_clip.load(
        "ViT-L/14",
        alpha_vision_ckpt_pth="/ailab/user/mahaoxuan/BIRD/clip_l14_grit1m_fultune_8xe.pth",
        device=device
    )
    model = model.float()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 设置混合精度训练
    scaler = GradScaler()

    # 掩码转换
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(0.5, 0.26)
    ])

    # SAM 模型路径
    sam_checkpoint = "/ailab/user/mahaoxuan/BIRD/sam_vit_b_01ec64.pth"
    model_type = 'vit_b'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)

    # 数据路径
    images_dir = "/ailab/user/mahaoxuan/BIRD/birdDataSet_mixed/bird/images"
    train_txt_path = "/ailab/user/mahaoxuan/BIRD/birdDataSet_mixed/bird/trainDataSet_mixed.txt"
    classes_txt_path = "/ailab/user/mahaoxuan/BIRD/birdDataSet_mixed/bird/classes.txt"
    
    bird = Bird(images_dir, train_txt_path, classes_txt_path, sam_checkpoint, model_type=model_type)
    bird.predictor = predictor

    dataset = BirdDataset(bird, preprocess, mask_transform)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=1, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    tokenizer = alpha_clip.tokenize

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        sampler.set_epoch(epoch)

        # 在主进程中显示进度条
        data_iter = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}]", disable=(rank != 0))

        for batch_idx, (image_tensor, alpha, class_names) in enumerate(data_iter):
            image_tensor = image_tensor.float().to(device)
            alpha = alpha.float().to(device)
            text_inputs = tokenizer(class_names).to(device)

            # 使用自动混合精度的前向传播
            with autocast():
                image_features = model.module.visual(image_tensor, alpha)
                text_features = model.module.encode_text(text_inputs)
                loss = clip_loss(image_features, text_features)

            optimizer.zero_grad()
            # 混合精度的反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if rank == 0:
                data_iter.set_postfix({'Batch Loss': loss.item()})

        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Average Loss: {avg_loss:.4f}")
            save_path = f"./models/alpha_clip_epoch_{epoch+1}.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            torch.save(model.state_dict(), "./models/alpha_clip_last.pth")

        # 清理缓存以减少显存占用
        torch.cuda.empty_cache()

    dist.destroy_process_group()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Distributed training script using multiple GPUs.")
    parser.add_argument('--gpu_numbers', type=int, default=4, help="Number of GPUs to use.")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()
    args.world_size = args.gpu_numbers

    mp.spawn(train, nprocs=args.gpu_numbers, args=(args,))

if __name__ == "__main__":
    main()
