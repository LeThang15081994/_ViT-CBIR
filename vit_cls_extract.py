import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
import cv2
import pickle
import os
import logging
from datetime import datetime


class vit_feature_extract:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(self.device)

        log_directory = 'D:/WorkSpace/_thangle15894/_AIproject/_ImageSearch/_ViT-CBIR/logging'
        today = datetime.now().strftime('%Y_%m_%d')  
        daily_directory = os.path.join(log_directory, today) 
        os.makedirs(daily_directory, exist_ok=True)

        log_file = os.path.join(daily_directory, 'logging_error.txt')
        logging.basicConfig(level=logging.INFO,  
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=log_file,  
                            filemode='a')  
        self.logger = logging.getLogger(__name__)  

    def vit_preprocessing(self, img):
        """Tiền xử lý một hình ảnh cho mô hình ViT và trích xuất token CLS.

        Args:
            img (Any): Hình ảnh đầu vào cần tiền xử lý.

        Returns:
            Output: Đại diện token CLS hoặc None nếu xảy ra lỗi.
        """
        try:
            print('Đang xử lý-------------------------------------------------------------------------------')
            inputs = self.processor(img, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states = True).hidden_states[-1][:, 0 ,:].detach().cpu().numpy() # get CLS token
            print('Hoàn tất-------------------------------------------------------------------------------')
            return outputs
        except Exception as e:
            self.logger.error(f"error in processing: {e}")
            return None

    def store_vit_CLS(self, data_path):
        """Lưu trữ các CLS tocken.

        Args:
            img (Any): Hình ảnh đầu vào cần tiền xử lý và trích xuất CLS.

        Returns:
            Output: cls_data.pkl hoặc none nếu xảy ra lỗi.
        """
        src_image = []
        paths = []
        try: 
            for img_path in os.listdir(data_path):
                img_path_full = os.path.join(data_path, img_path)
                img = cv2.imread(img_path_full)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                src_image.append(img)
                paths.append(img_path_full)

            cls_vectors = self.vit_preprocessing(src_image)
            
            with open(os.path.join(data_path, 'cls.pkl'), 'wb') as f:
                pickle.dump(cls_vectors, f)
            self.logger.info(f"Saved {len(cls_vectors)} CLS token.")
        except Exception as e:
            self.logger.error(f"error in processing - Archive failed: {e}")
            return None


if __name__ == "__main__":
    img_preprocess = vit_feature_extract()
    img_preprocess.store_vit_CLS('./dataset/test_images')
   