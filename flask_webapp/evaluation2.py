import os
import numpy as np
import tensorflow as tf
import cv2
from scipy.linalg import sqrtm
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from PIL import Image
import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import h5py


# Định nghĩa kích thước ảnh đầu vào cho generator
IMG_SIZE = 256


def preprocess_test_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Đảm bảo ảnh có 3 kênh
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype(np.float32)
    image = (image / 127.5) - 1  # Chuẩn hóa ảnh về khoảng [-1, 1]
    return image


def generate_image(generator, input_image):
    # Thay đổi kích thước của ảnh đầu vào
    input_image_resized = tf.image.resize(input_image, (256, 256))
    input_image_resized = tf.expand_dims(input_image_resized, 0)  # Thêm chiều batch
    generated_image = generator(input_image_resized, training=False)[0]
    return generated_image



def calculate_fid_pix2pix(real_activations, fake_activations):
    # Calculate mean and covariance statistics
    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_activations, axis=0), np.cov(fake_activations, rowvar=False)

    # Calculate squared difference between means
    diff_mean = np.sum((mu_real - mu_fake) ** 2)

    # Calculate squared root of matrix product of covariances
    cov_mean = sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    # Calculate FID
    fid = diff_mean + np.trace(sigma_real + sigma_fake - 2 * cov_mean)

    return fid


def evaluate_generator_FID(generator, test_images):
    # Load InceptionV3 model pretrained on ImageNet
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=[299, 299, 3])

    real_activations = []
    fake_activations = []


    # Tính toán kích hoạt cho ảnh thật
    for image in test_images:
        original_image_resized = tf.image.resize(image, (299, 299))
        expanded_image = np.expand_dims(original_image_resized, axis=0).copy()  # Tạo bản sao trước khi thay đổi
        real_activations.append(inception_model.predict(preprocess_input(expanded_image)))

    real_activations = np.concatenate(real_activations, axis=0)

    # Tính toán kích hoạt cho ảnh giả
    for image in test_images:
        generated_image = generate_image(generator, image)
        expanded_image = np.expand_dims(generated_image, axis=0).copy()  # Tạo bản sao trước khi thay đổi
        fake_activations.append(inception_model.predict(preprocess_input(expanded_image)))

    fake_activations = np.concatenate(fake_activations, axis=0)

    # Calculate FID
    fid = calculate_fid_pix2pix(real_activations, fake_activations)

    return fid


def load_images_from_folder(folder_path):
    images = []
    count = 0
    for filename in os.listdir(folder_path):
        if count >= 100 or count == len(os.listdir(folder_path)):
            break
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            image = preprocess_test_image(img_path)
            images.append(image)
            count += 1
    return images



# Load generator model
generator_file = 'D:/KHMT_2020604284_DoTrungPhong/flask_webapp/model/pix2pix_generator/cgan_generator.h5'
generator = keras.models.load_model(generator_file)
test_folder = 'C:/Users/DELL/Pictures/fake'
test_images = load_images_from_folder(test_folder)
fid = evaluate_generator_FID(generator, test_images)
print("Fréchet Inception Distance (FID):", fid)
