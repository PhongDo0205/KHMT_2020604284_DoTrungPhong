import sqlite3
import os



def read_data_from_table(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    return rows



database_name = os.path.join("D:/KHMT_2020604284_DoTrungPhong/flask_webapp", 'imageweb.db')
# Kết nối đến cơ sở dữ liệu
conn = sqlite3.connect(database_name)

# Đọc dữ liệu từ bảng users
users = read_data_from_table(conn, 'users')
print("Users:")
for user in users:
    print(user)

# Đọc dữ liệu từ bảng collections
collections = read_data_from_table(conn, 'collections')
print("\nCollections:")
for collection in collections:
    print(collection)

# Đọc dữ liệu từ bảng images
images = read_data_from_table(conn, 'images')
print("\nImages:")
for image in images:
    print(image)

# Đọc dữ liệu từ bảng tags
tags = read_data_from_table(conn, 'tags')
print("\nTags:")
for tag in tags:
    print(tag)

# Đọc dữ liệu từ bảng image_tags
image_tags = read_data_from_table(conn, 'image_tags')
print("\nImage Tags:")
for image_tag in image_tags:
    print(image_tag)

# Đóng kết nối
conn.close()
