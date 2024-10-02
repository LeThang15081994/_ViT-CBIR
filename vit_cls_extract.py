import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
import cv2
import pickle
import os


class vit_feature_extract:
    def __init__(self):
        pass

    def vit_preprocess(self, img): # image preprocessing convert to tensor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)

        inputs = processor(img, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs, output_hidden_state = True).hidden_states[-1][:,0,:].detach().cpu().numpy()

        return output

    def store_vit_CLS(self, data_path):  # new method to store vectors
        src_image = []
        paths = []

        for img_path in os.listdir(data_path):
            img_path_full = os.path.join(data_path, img_path)
            img = cv2.imread(img_path_full)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            src_image.append(img)
            paths.append(img_path_full)

if __name__ == "__main__":