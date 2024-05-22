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
    ('/static/temp_folder/5bb815071a2ff76093fa4f594c8fdfb1.jpg', '/static/predict_folder/generated_5bb815071a2ff76093fa4f594c8fdfb1.jpg', 'Image 1', 1),
    ('/static/temp_folder/399c37ae170e8c329a9bd623d5e4a2f6.jpg', '/static/predict_folder/generated_399c37ae170e8c329a9bd623d5e4a2f6.jpg', 'Image 2', 1),
    ('/static/temp_folder/adolf-hitler_castle-battlements.jpg', '/static/predict_folder/generated_adolf-hitler_castle-battlements.jpg', 'Image 3', 2)
]
cursor.executemany("INSERT INTO images (original_image_path, generated_image_path, title, collection_id) VALUES (?, ?, ?, ?)", images_data)

# Tạo dữ liệu mẫu cho bảng tags
tags_data = [
    ('Tag 1',),
    ('Tag 2',),
    ('Tag 3',)
]
cursor.executemany("INSERT INTO tags (name) VALUES (?)", tags_data)

# Tạo dữ liệu mẫu cho bảng image_tags (kết hợp ảnh và tag)
image_tags_data = [
    (1, 1),
    (1, 2),
    (2, 2),
    (3, 3)
]
cursor.executemany("INSERT INTO image_tags (image_id, tag_id) VALUES (?, ?)", image_tags_data)

# Lưu thay đổi và đóng kết nối
conn.commit()
conn.close()


print("Thêm cơ sở dữ liệu mới thành công!")