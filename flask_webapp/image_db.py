import sqlite3
import os
import requests
from werkzeug.utils import secure_filename

database_name = os.path.join("D:/KHMT_2020604284_DoTrungPhong/flask_webapp", 'imageweb.db')
schema_file = os.path.join("D:/KHMT_2020604284_DoTrungPhong/flask_webapp", 'create_database.sql')

def initialize_database(db_name, schema_file):
    """
    Hàm này tạo cơ sở dữ liệu từ file schema.sql.

    :param db_name: Tên của cơ sở dữ liệu SQLite sẽ được tạo hoặc kết nối.
    :param schema_file: Tên của file .sql chứa các lệnh tạo bảng.
    """
    # Kết nối đến cơ sở dữ liệu (tạo mới nếu chưa có)
    conn = sqlite3.connect(db_name)

    try:
        # Đọc nội dung của file schema.sql
        with open(schema_file, 'r', encoding='utf-8') as sql_file:
            sql_script = sql_file.read()

        # Tạo con trỏ để thực hiện các lệnh SQL
        cursor = conn.cursor()

        # Thực thi các lệnh SQL trong file schema.sql
        cursor.executescript(sql_script)

        # Lưu thay đổi
        conn.commit()
        print("Cơ sở dữ liệu đã được khởi tạo thành công.")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
        conn.rollback()
    finally:
        # Đóng kết nối
        conn.close()




# Gọi hàm để khởi tạo cơ sở dữ liệu
initialize_database(database_name, schema_file)



