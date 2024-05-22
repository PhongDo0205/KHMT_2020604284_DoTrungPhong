import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os

model_dir = "D:/KHMT_2020604284_DoTrungPhong/flask_webapp/model/pix2pix_generator"
# Đọc kiến trúc mô hình từ tệp JSON
json_path = os.path.join(model_dir, "model_architecture.json")
with open(json_path, "r") as json_file:
    model_json = json_file.read()

# Tạo mô hình từ kiến trúc JSON
model = model_from_json(model_json)

# Tải trọng số mô hình từ tệp HDF5
weights_path = os.path.join(model_dir, "model_weights.weights.h5")
model.load_weights(weights_path)
print("load thành công")
# Bây giờ bạn có thể sử dụng mô hình để dự đoán hoặc sinh ảnh
# Ví dụ: Sinh ảnh từ noise vector ngẫu nhiên
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def preprocess_test_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Đảm bảo ảnh có 3 kênh
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype(np.float32)
    image = (image / 127.5) - 1  # Chuẩn hóa ảnh về khoảng [-1, 1]
    return image


def generate_image(generator, input_image):
    input_image = tf.expand_dims(input_image, 0)  # Thêm batch dimension
    generated_image = generator(input_image, training=False)[0]  # Chuyển đổi từ [-1, 1] về [0, 1]
    generated_image = (generated_image + 1) / 2
    return generated_image



def show_and_save_images(original_image, generated_image, save_path):
    plt.figure(figsize=(10, 5))
    
    # Hiển thị ảnh gốc
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow((original_image + 1) / 2)  # Chuyển đổi từ [-1, 1] về [0, 1]
    plt.axis('off')

    # Hiển thị ảnh sinh ra
    plt.subplot(1, 2, 2)
    plt.title("Generated Image")
    plt.imshow(generated_image)  # Ảnh đã ở khoảng [0, 1]
    plt.axis('off')

    plt.show()

    # Lưu ảnh sinh ra
    generated_image = (generated_image * 255).numpy().astype(np.uint8)
    Image.fromarray(generated_image).save(save_path)



def load_show_and_save_images(generator, test_folder, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for image_name in os.listdir(test_folder):
        image_path = os.path.join(test_folder, image_name)
        
        # Đọc và tiền xử lý ảnh đầu vào
        original_image = preprocess_test_image(image_path)
        
        # Sinh ảnh từ model generator
        generated_image = generate_image(generator, original_image)
        
        # Hiển thị và lưu ảnh
        save_path = os.path.join(save_folder, 'generated_' + image_name)
        show_and_save_images(original_image, generated_image, save_path)



IMG_SIZE = 256

test_folder = "C:/Users/DELL/Pictures/fake"
save_folder = "C:/Users/DELL/Pictures/out"

load_show_and_save_images(model, test_folder, save_folder)

