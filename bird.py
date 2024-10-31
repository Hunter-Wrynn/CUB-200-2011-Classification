# bird.py

import numpy as np
from PIL import Image
import os
from segment_anything import sam_model_registry, SamPredictor

class Bird:
    def __init__(self, images_dir, train_txt_path, sam_checkpoint, model_type='vit_b'):
        """
        初始化 Bird 对象，加载数据集和 SAM 模型。

        参数：
        - images_dir (str): 图像目录的路径。
        - train_txt_path (str): 'trainDataSet_mixed.txt' 文件的路径。
        - sam_checkpoint (str): SAM 模型检查点文件的路径。
        - model_type (str): 使用的 SAM 模型类型（'vit_h', 'vit_l', 'vit_b'）。
        """
        self.images_dir = images_dir
        self.train_txt_path = train_txt_path
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.data = self.load_dataset()
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.predictor = SamPredictor(self.sam)

    def load_dataset(self):
        """
        从 'trainDataSet_mixed.txt' 文件中加载数据集。

        返回：
        - data (list of dict): 包含图像信息的列表，每个元素包含 'image_id'，'image_path'，'bbox'。
        """
        data = []
        with open(self.train_txt_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) >= 6:
                    image_id = tokens[0]
                    image_path_tokens = tokens[1:-4]
                    image_path = ' '.join(image_path_tokens)
                    bbox = list(map(int, tokens[-4:]))
                else:
                    image_id = tokens[0]
                    image_path = tokens[1]
                    bbox = list(map(int, tokens[2:6]))

                data.append({
                    'image_id': image_id,
                    'image_path': image_path,
                    'bbox': bbox
                })
        return data

    def load_image(self, image_path):
        """
        加载指定路径的图像。

        参数：
        - image_path (str): 相对于 images_dir 的图像路径。

        返回：
        - image (PIL.Image): 加载的图像。
        """
        full_image_path = os.path.join(self.images_dir, image_path)
        image = Image.open(full_image_path).convert("RGB")
        return image

    def generate_mask(self, image, bbox):
        """
        使用 SAM 模型生成图像的掩码。

        参数：
        - image (PIL.Image): 图像对象。
        - bbox (list): 边界框坐标 [x1, y1, x2, y2]。

        返回：
        - mask (numpy.ndarray): 生成的掩码。
        """
        image_np = np.array(image)
        self.predictor.set_image(image_np)

        input_box = np.array(bbox)[None, :]  # 形状为 (1, 4)

        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )

        return masks[0]

    def get_mask_by_image_id(self, image_id):
        """
        根据图像编号获取对应的掩码。

        参数：
        - image_id (str): 图像编号。

        返回：
        - mask (numpy.ndarray): 生成的掩码。
        - image (PIL.Image): 图像对象。
        """
        for item in self.data:
            if item['image_id'] == image_id:
                image = self.load_image(item['image_path'])
                mask = self.generate_mask(image, item['bbox'])
                return mask, image
        return None, None
