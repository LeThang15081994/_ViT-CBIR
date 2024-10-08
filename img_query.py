import matplotlib.pyplot as plt
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
import cv2
import pickle
import math
from PIL import Image

from vit_cls_extract import vit_feature_extract

def visualization(vectors,img_query, paths):
    ids = np.argsort(vectors)[:20] # get 20 image have nearest image.
    nearest_image = [(paths[id], vectors[id]) for id in ids]
    # visualization
    axes = []
    grid_size = int(math.ceil(math.sqrt(len(nearest_image))))
    fig = plt.figure(figsize=(10,8))
    #plt.title('Hình ảnh search')

    ax_query = fig.add_subplot(grid_size, grid_size, 1)
    ax_query.set_title('Hình ảnh Query')
    ax_query.imshow(img_query)
    ax_query.axis('off')  # Ẩn trục

    for id in range(len(nearest_image)):
        draw_image = nearest_image[id]
        axes.append(fig.add_subplot(grid_size, grid_size, id+2))

        axes[-1].set_title(f'{draw_image[1]:.2f}')
        plt.imshow(Image.open(draw_image[0]))

    fig.tight_layout()
    plt.show()

def L2_norm(preprocess_data, preprocess_query):
    L2 = np.linalg.norm(preprocess_data - preprocess_query, axis= 1)
    return L2
    

if __name__ == "__main__":
    #Load hình ảnh
    img_preprocess = cv2.imread('./testimg/wolf2.jpg')
    img_preprocess = cv2.cvtColor(img_preprocess,cv2.COLOR_BGR2RGB)

    #Tiền xử lý hình ảnh chuyển thành vector
    preprocess_img = vit_feature_extract()
    img_query = preprocess_img.vit_preprocessing(img_preprocess)

    
    #load data vector
    with open("./database/cls_vetors.pkl", "rb") as f:
        vectors = np.array(pickle.load(f))[:, 0, :]
    with open("./database/paths_vetors.pkl", "rb") as f:
        paths = pickle.load(f)
    
    # tính khoảng cách sử dụng L2
    distance = L2_norm(vectors, img_query)
    visualization(distance, img_preprocess, paths)
    