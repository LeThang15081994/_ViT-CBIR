import os

# Đường dẫn đến thư mục chứa các hình ảnh
folder_path = './dataset/train_images'  # Thay thế bằng đường dẫn thư mục thực tế

# Lấy danh sách các file hình ảnh trong thư mục
images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Sắp xếp danh sách hình ảnh (tùy chọn)
images.sort()

# Đổi tên các hình ảnh
for index, image_name in enumerate(images):
    # Tạo tên mới
    new_name = f'image_{index + 1:03d}.jpg'  # Đổi theo định dạng image_001.jpg
    # Đường dẫn cũ và mới
    old_path = os.path.join(folder_path, image_name)
    new_path = os.path.join(folder_path, new_name)
    
    # Đổi tên file
    os.rename(old_path, new_path)

print("Đã đổi tên các hình ảnh thành công!")