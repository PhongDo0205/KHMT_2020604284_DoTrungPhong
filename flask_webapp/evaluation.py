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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def preprocess_images(images):
    images_resized = np.array([np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((299, 299))) for img in images])
    images_preprocessed = preprocess_input(images_resized)
    return images_preprocessed

def inception_score(images, batch_size=32):
    inception_model = InceptionV3(include_top=True, weights='imagenet')
    processed_images = preprocess_images(images)
    preds = inception_model.predict(processed_images, batch_size=batch_size)
    preds = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
    scores = np.sum(preds * np.log(preds), axis=1)
    scores = np.exp(np.mean(scores))
    return scores

def calculate_fid(real_images, generated_images):
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    real_images = preprocess_images(real_images)
    generated_images = preprocess_images(generated_images)
    real_activations = inception_model.predict(real_images)
    generated_activations = inception_model.predict(generated_images)
    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_gen, sigma_gen = np.mean(generated_activations, axis=0), np.cov(generated_activations, rowvar=False)
    diff_mean = np.sum((mu_real - mu_gen)**2)
    cov_mean = sqrtm(sigma_real.dot(sigma_gen))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid = diff_mean + np.trace(sigma_real + sigma_gen - 2*cov_mean)
    return fid

def generate_fake_samples(generator, seed_size, n_samples):
    generated_images = []
    for _ in range(n_samples):
        random_seed = tf.random.normal([1, seed_size])
        new_image = generator(random_seed, training=False)
        generated_images.append(new_image[0].numpy())
    return generated_images

def evaluate_GAN(generator, seed_size, n_samples, real_images):
    fake_samples = generate_fake_samples(generator, seed_size, n_samples)
    score = inception_score(fake_samples)
    fid = calculate_fid(real_images, fake_samples)
    return score, fid

def load_images_from_directory(image_dir):
    images = []
    count = 0
    for filename in os.listdir(image_dir):
        if count >= 150:  # Số lượng ảnh tối đa
            break
        img_path = os.path.join(image_dir, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB
            img = img / 255.0  # Chuẩn hóa giá trị ảnh từ 0-255 về 0-1
            images.append(img)
            count += 1
    return images


model_folder = "D:/KHMT_2020604284_DoTrungPhong/flask_webapp/model"
# dcgan = keras.models.load_model(os.path.join("D:/KHMT_2020604284_DoTrungPhong/flask_webapp/model", 'dcgan_generator.h5'))

# Cố gắng tải mô hình với compile=False
try:
    dcgan = keras.models.load_model(os.path.join(model_folder, 'dcgan_generator.h5'), compile=False)
    print("Tải mô hình thành công với compile=False")
except Exception as e:
    print(f"Lỗi khi tải mô hình với compile=False: {e}")

# Nếu trên không thành công, thử tải mô hình bình thường
try:
    dcgan = keras.models.load_model(os.path.join(model_folder, 'dcgan_generator.h5'))
    print("Tải mô hình thành công")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")

# Xác minh mô hình đã tải bằng cách in ra summary
try:
    dcgan.summary()
except Exception as e:
    print(f"Lỗi khi hiển thị tóm tắt mô hình: {e}")

# generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# dcgan.compile(loss='binary_crossentropy', optimizer=generator_optimizer)


seed_size = 123

validation_image_dir = "D:/KHMT_2020604284_DoTrungPhong/portrait_painting"
n_samples = 150
real_images = load_images_from_directory(validation_image_dir)
score, fid = evaluate_GAN(dcgan, seed_size, n_samples, real_images)
print("Inception Score:", score)
print("Fréchet Inception Distance:", fid)



