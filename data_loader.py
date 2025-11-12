import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import Dict, Any

import torchvision.transforms.functional as TF
import random
from PIL import Image, ImageFilter
import io

# ====================== 自定义增强模块 ======================
class RandomJPEGCompression:
    """随机 JPEG 压缩失真"""
    def __init__(self, quality_range=(30, 90)):
        self.quality_range = quality_range
    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

class RandomOcclusion:
    """随机遮挡部分区域"""
    def __init__(self, size_range=(0.1, 0.3), p=0.3):
        self.size_range = size_range
        self.p = p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        occ_w = int(random.uniform(*self.size_range) * w)
        occ_h = int(random.uniform(*self.size_range) * h)
        x1 = random.randint(0, w - occ_w)
        y1 = random.randint(0, h - occ_h)
        img = img.copy()
        img.paste((0, 0, 0), (x1, y1, x1 + occ_w, y1 + occ_h))
        return img

# ====================== 数据增强定义 ======================
def build_transforms(train=True, crop_size=(112, 112), grayscale=False, aug_type="standard"):
    t = []
    if grayscale:
        t.append(transforms.Grayscale(num_output_channels=3))
    t.append(transforms.Resize(crop_size))

    if train:
        if aug_type == "standard":
            t += [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ]
        elif aug_type == "strong":
            t += [
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.02),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=(-20, 20), translate=(0.12, 0.12), scale=(0.85, 1.15)),
                transforms.RandomGrayscale(p=0.2),
                transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.2))) if random.random() < 0.4 else img),
                RandomJPEGCompression(quality_range=(30, 80)),
                RandomOcclusion(size_range=(0.12, 0.24), p=0.28),
                transforms.Lambda(lambda img: TF.adjust_sharpness(img, sharpness_factor=random.uniform(0.8, 1.8)) if random.random() < 0.4 else img),
            ]
    else:
        t.append(transforms.CenterCrop(crop_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(
        mean=[0.3193, 0.2874, 0.2578],
        std=[0.1980, 0.2076, 0.2109]
    ))
    return transforms.Compose(t)

# ====================== 数据集定义 ======================
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, test_size=0.5, random_state=42, grayscale=False):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths, self.labels = [], []
        self.label_map = {}
        self.train = train

        # 过滤掉样本少于 2 张的类
        self.classes = []
        for c in sorted(os.listdir(root_dir)):
            c_path = os.path.join(root_dir, c)
            if not os.path.isdir(c_path):
                continue
            num_imgs = len([f for f in os.listdir(c_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if num_imgs < 2:
                print(f"⚠️ Skipping class '{c}' (only {num_imgs} image).")
                continue
            self.classes.append(c)

        # 建立样本路径和标签
        for idx, student in enumerate(self.classes):
            self.label_map[idx] = student
            student_dir = os.path.join(root_dir, student)
            for img_file in os.listdir(student_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.img_paths.append(os.path.join(student_dir, img_file))
                    self.labels.append(idx)

        # 如果只有 1 类或样本太少，直接抛错
        if len(set(self.labels)) < 2:
            raise RuntimeError(f"Too few valid classes after filtering in {root_dir}")

        # train/test split，若 stratify 无法执行则降级
        try:
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                self.img_paths, self.labels, test_size=test_size, stratify=self.labels, random_state=random_state
            )
        except ValueError as e:
            print(f"⚠️ Stratified split failed: {e}\nUsing random split instead.")
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                self.img_paths, self.labels, test_size=test_size, random_state=random_state
            )

        if self.train:
            self.img_paths, self.labels = train_paths, train_labels
        else:
            self.img_paths, self.labels = test_paths, test_labels

        self.meta = {
            "num_classes": len(self.classes),
            "num_images": len(self.img_paths),
            "class_names": self.classes,
            "grayscale": grayscale,
            "transform": repr(transform),
            "test_size": test_size,
        }

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')
        if self.meta["grayscale"]:
            img = img.convert('L').convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, label

# ====================== DataLoader 构建 ======================
def get_data_loader(root_dir, train=True, crop_size=(112, 112), test_size=0.5, batch_size=32,
                    shuffle=True, num_workers=4, grayscale=False, aug_type="standard"):
    """
    Build DataLoader, automatically skip classes with fewer than 2 images.
    Returns: data_loader, snapshot_meta_dict
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")

    transform = build_transforms(train=train, crop_size=crop_size, grayscale=grayscale, aug_type=aug_type)
    dataset = FaceDataset(root_dir=root_dir, transform=transform, train=train, test_size=test_size, grayscale=grayscale)
    print(f"✅ Loaded {dataset.meta['num_classes']} valid classes (≥2 images each).")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    snapshot_meta = dict(
        num_classes=dataset.meta["num_classes"],
        class_names=dataset.meta["class_names"],
        grayscale=grayscale,
        crop_size=crop_size,
        test_size=test_size,
        aug_type=aug_type,
        transform=repr(transform),
        batch_size=batch_size,
    )
    return data_loader, snapshot_meta

# ====================== 调试 & 可视化 ======================
if __name__ == "__main__":
    root_dir = r"./data"
    grayscale = False
    aug_type = "standard"   # 可选: 'standard', 'strong'
    crop_size = (112, 112)

    train_loader, train_meta = get_data_loader(root_dir, train=True, crop_size=crop_size,
                                               batch_size=4, shuffle=True, grayscale=grayscale, aug_type=aug_type)
    print(f"Train snapshot meta: {train_meta}")

    imgs, labels = next(iter(train_loader))
    print(f"Batch: {imgs.shape}, Labels: {labels}")
    flag = "train" if train_loader.dataset.train else "test"

    # ---------- 反归一化后可视化 ----------
    mean = np.array([0.3193, 0.2874, 0.2578])
    std = np.array([0.1980, 0.2076, 0.2109])

    fig, axes = plt.subplots(1, imgs.shape[0], figsize=(12, 3))
    for i in range(imgs.shape[0]):
        img_np = imgs[i].permute(1, 2, 0).numpy()
        img_np = (img_np * std + mean).clip(0, 1)  # 反归一化
        axes[i].imshow(img_np)
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(f"faces_visualized_{flag}.png")
    plt.show()
