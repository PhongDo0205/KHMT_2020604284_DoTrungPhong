import sqlite3
from werkzeug.security import generate_password_hash
import os


# Kết nối đến cơ sở dữ liệu
database_name = os.path.join("D:/KHMT_2020604284_DoTrungPhong/flask_webapp", 'imageweb.db')
conn = sqlite3.connect(database_name)
cursor = conn.cursor()

# Tạo dữ liệu mẫu cho bảng users
users_data = [
    ('user1', generate_password_hash('password1'), 'user1@example.com'),
    ('user2', generate_password_hash('password2'), 'user2@example.com'),
    ('user3', generate_password_hash('password3'), 'user3@example.com')
]
cursor.executemany("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", users_data)

# Tạo dữ liệu mẫu cho bảng collections
collections_data = [
    ('Collection 1', 1),
    ('Collection 2', 1),
    ('Collection 3', 2)
]
cursor.executemany("INSERT INTO collections (name, user_id) VALUES (?, ?)", collections_data)

# Tạo dữ liệu mẫu cho bảng images
images_data = [
    ('/static/random_image_folder/generated_3d32b0f5.jpg', '/static/random_image_folder/generated_3d32b0f5.jpg', 'Portrait 1', 1),
    ('/static/random_image_folder/generated_7bfd6c6d.jpg', '/static/random_image_folder/generated_7bfd6c6d.jpg', 'Portrait 2', 1),
    ('/static/random_image_folder/generated_93b2954c.jpg', '/static/random_image_folder/generated_93b2954c.jpg', 'Portrait 3', 1),
    ('/static/random_image_folder/generated_e675df78.jpg', '/static/random_image_folder/generated_e675df78.jpg', 'Portrait 4', 1)
]
cursor.executemany("INSERT INTO images (original_image_path, generated_image_path, title, collection_id) VALUES (?, ?, ?, ?)", images_data)

# Tạo dữ liệu mẫu cho bảng tags
tags_data = [
    ('Portrait Image',),
    ('Sketch to Image',)
]
cursor.executemany("INSERT INTO tags (name) VALUES (?)", tags_data)

# Tạo dữ liệu mẫu cho bảng image_tags (kết hợp ảnh và tag)
image_tags_data = [
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1)
]
cursor.executemany("INSERT INTO image_tags (image_id, tag_id) VALUES (?, ?)", image_tags_data)

# Lưu thay đổi và đóng kết nối
conn.commit()
conn.close()


print("Thêm cơ sở dữ liệu mới thành công!")