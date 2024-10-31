import numpy as np
from PIL import Image
import os
from segment_anything import sam_model_registry, SamPredictor

class Bird:
    def __init__(self, images_dir, train_txt_path, classes_txt_path, sam_checkpoint, model_type='vit_b'):
        """
        初始化 Bird 对象，加载数据集和 SAM 模型。
        """
        self.images_dir = images_dir
        self.train_txt_path = train_txt_path
        self.classes_txt_path = classes_txt_path
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.class_id_to_name = self.load_classes()
        self.data = self.load_dataset()
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.predictor = SamPredictor(self.sam)

    def load_classes(self):
        """
        加载类别编号和类别名称的映射关系。
        """
        class_id_to_name = {}
        with open(self.classes_txt_path, 'r') as f:
            for line in f:
                tokens = line.strip().split(maxsplit=1)
                if len(tokens) == 2:
                    class_id = tokens[0]  # '1', '2', ...
                    class_name = tokens[1]  # '001.Black_footed_Albatross', ...
                    class_id_to_name[class_id] = class_name
        return class_id_to_name

    def load_dataset(self):
        """
        从 'trainDataSet_mixed.txt' 文件中加载数据集。

        返回：
        - data (list of dict): 包含图像信息的列表，每个元素包含 'image_id'，'image_path'，'bbox'，'class_name'。
        """
        data = []
        with open(self.train_txt_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) >= 6:
                    image_id = tokens[0]
                    image_path = tokens[1]
                    bbox = list(map(int, tokens[2:6]))
                else:
                    continue  # 跳过格式不正确的行

                # 提取类别编号
                class_folder = image_path.split('/')[0]  # '001.Black_footed_Albatross'
                class_id_str = class_folder.split('.')[0]  # '001'
                class_id = str(int(class_id_str.lstrip('0')))  # 去掉前导零并转换为字符串

                class_name = self.class_id_to_name.get(class_id, 'Unknown')

                # 格式化 class_name，去掉编号和下划线
                if class_name != 'Unknown':
                    class_name = ' '.join(class_name.split('.')[1:]).replace('_', ' ')
                
                data.append({
                    'image_id': image_id,
                    'image_path': image_path,
                    'bbox': bbox,
                    'class_name': class_name
                })
        return data


    def load_image(self, image_path):
        """
        加载指定路径的图像。
        """
        full_image_path = os.path.join(self.images_dir, image_path)
        image = Image.open(full_image_path).convert("RGB")
        return image

    def generate_mask(self, image, bbox):
        """
        使用 SAM 模型生成图像的掩码。
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

    def get_item_by_index(self, index):
        """
        根据索引获取数据项，包括图像、掩码和类别名称。
        """
        item = self.data[index]
        image = self.load_image(item['image_path'])
        mask = self.generate_mask(image, item['bbox'])
        class_name = item['class_name']
        return image, mask, class_name
