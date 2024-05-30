import sqlite3
import os
import numpy as np
from flask import Flask, render_template, redirect, session, request, flash, jsonify, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import keras
import uuid
from io import BytesIO
from PIL import Image
import cv2
from tf_keras.models import load_model
import h5py

app = Flask(__name__)
app.secret_key = 'anything'

database_name = os.path.join(os.path.dirname(__file__), 'imageweb.db')
model_folder = os.path.join(os.path.dirname(__file__), 'model')
static_folder = os.path.abspath("D:/KHMT_2020604284_DoTrungPhong/flask_webapp/static")
print(database_name)


# Hàm load model
def load_cgan_model(model_dir, json_file="model_architecture.json", weights_file="model_weights.weights.h5"):
    # Đường dẫn tới tệp JSON và HDF5
    json_path = os.path.join(model_dir, json_file)
    weights_path = os.path.join(model_dir, weights_file)

    # Đọc kiến trúc mô hình từ tệp JSON
    with open(json_path, "r") as json_file:
        model_json = json_file.read()

    # Tạo mô hình từ kiến trúc JSON
    model = keras.models.model_from_json(model_json)

    # Tải trọng số mô hình từ tệp HDF5
    model.load_weights(weights_path)

    return model


cgan_model_dir = "D:/KHMT_2020604284_DoTrungPhong/flask_webapp/model/pix2pix_generator"
cgan = load_cgan_model(cgan_model_dir)



dcgan_model_dir = "D:/KHMT_2020604284_DoTrungPhong/flask_webapp/model/dcgan_generator/dcgan_generator.h5"
dcgan = load_model(dcgan_model_dir)


# Hàm tạo ảnh trong csdl
def add_image(original_image_path, generated_image_path, title):
    try:
        conn = sqlite3.connect(database_name)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO images (original_image_path, generated_image_path, title)
        VALUES (?, ?, ?)
        ''', (original_image_path, generated_image_path, title))
        conn.commit()
        image_id = cursor.lastrowid
        return image_id
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        if conn:
            conn.close()

# Hàm tạo bộ sưu tập trong csdl
def add_collection(name, user_id):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO collections (name, user_id)
    VALUES (?, ?)
    ''', (name, user_id))
    conn.commit()
    conn.close()

# Hàm thêm ảnh vào bộ sưu tập trong csdl
def add_image_to_collection(image_id, collection_id):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute('''
    UPDATE images
    SET collection_id = ?
    WHERE id = ?
    ''', (collection_id, image_id))
    conn.commit()
    conn.close()


# Hàm lấy thông tin người dùng
def get_user_id(username):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return None

# Hàm kiểm tra nếu ảnh đã tồn tại
def get_image_id_by_path(generated_image_path):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM images WHERE generated_image_path = ?', (generated_image_path,))
    image = cursor.fetchone()
    conn.close()
    return image[0] if image else None

# Hàm lấy collection id
def get_collection_id(collection_name, user_id):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM collections WHERE name = ? AND user_id = ?', (collection_name, user_id))
    collection = cursor.fetchone()
    conn.close()
    return collection[0] if collection else None

# Hàm lấy tag id
def get_tag_id_by_image(image_id):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute('SELECT tag_id FROM image_tags WHERE image_id = ?', (image_id,))
    tag = cursor.fetchone()
    conn.close()
    return tag[0] if tag else None

# Hàm thêm tag vào ảnh
def add_tag_to_image(image_id, tag_id):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO image_tags (image_id, tag_id) VALUES (?, ?)', (image_id, tag_id))
    conn.commit()
    conn.close()









# Hàm thêm người dùng mới vào cơ sở dữ liệu
def add_new_user(username, password, email):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    hashed_password = generate_password_hash(password)  # Tạo hash cho mật khẩu
    cursor.execute('''
    INSERT INTO users (username, password, email)
    VALUES (?, ?, ?)
    ''', (username, hashed_password, email))
    conn.commit()
    conn.close()

# Hàm kiểm tra thông tin người dùng khi đăng nhập
def check_user_info(username, password):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    if result and check_password_hash(result[0], password):
        return True
    return False




# Hàm tạo ảnh ngẫu nhiên
def display_generated_image(generator, seed_size):
    noise = np.random.normal(0, 1, size=(1, seed_size))
    new_image = generator(noise, training=False)
    generated_image = (new_image * 0.5 + 0.5) * 255
    generated_image = generated_image.astype(np.uint8)
    return generated_image



# Tạo ảnh ngãu nhiên
@app.route('/generate_random', methods=['POST'])
def generate_random():
    SEED_SIZE = 100
    noise = tf.random.normal([1, SEED_SIZE])
    generated_image = dcgan(noise, training=False)
    generated_image = 0.5 * generated_image + 0.5
    generated_image = (generated_image * 255).numpy().astype(np.uint8)

    generated_image = np.squeeze(generated_image, axis=0)
    if generated_image.shape[-1] == 1:  # Handle grayscale images if needed
        generated_image = np.squeeze(generated_image, axis=-1)

    random_string = str(uuid.uuid4())[:8]

    generated_image_folder = os.path.join(static_folder, "random_image_folder")
    os.makedirs(generated_image_folder, exist_ok=True)  # Ensure the folder exists
    generated_image_filename = f"generated_{random_string}.jpg"
    generated_image_path = os.path.join(generated_image_folder, generated_image_filename)

    # Convert the numpy array to an Image object and save it
    generated_image_pil = Image.fromarray(generated_image)
    generated_image_pil.save(generated_image_path)
    
    # add_image(generated_image_path, generated_image_path)
    return jsonify({'image_path': generated_image_path.replace(static_folder, "/static").replace("\\", "/")})


# Hiển thị ảnh
@app.route('/image_page')
def show_generated_image():
    logged_in = session.get('logged_in', False)
    image_path = request.args.get('image_path')
    return render_template('image_page.html', image_path=image_path, logged_in=logged_in)






# Tiền xử lý ảnh
def preprocess_test_image(image_path):
    IMG_SIZE = 256
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32)
    image = (image / 127.5) - 1
    return image


# Tạo ảnh 
def generate_image(generator, input_image):
    input_image = np.expand_dims(input_image, axis=0)
    generated_image = generator.predict(input_image)[0]
    generated_image = (generated_image + 1) / 2
    return generated_image


# Tạo ảnh chi tiết từ ảnh phác hạo
@app.route('/generate_from_sketch', methods=['POST'])
def generate_from_sketch():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Generate a random string to avoid filename collision
    random_string = str(uuid.uuid4())[:8]

    original_image_folder = os.path.join(static_folder, "temp_folder")
    os.makedirs(original_image_folder, exist_ok=True)  # Ensure the folder exists
    filename, file_extension = os.path.splitext(file.filename)
    original_image_path = os.path.join(original_image_folder, f"original_{random_string}_{filename}{file_extension}")
    file.save(original_image_path)
    original_image = preprocess_test_image(original_image_path)

    generated_image_folder = os.path.join(static_folder, "predict_folder")
    os.makedirs(generated_image_folder, exist_ok=True)  # Ensure the folder exists
    generated_image_path = os.path.join(generated_image_folder, f"generated_{random_string}_{filename}{file_extension}")

    generated_image = generate_image(cgan, original_image)
    generated_image_pil = Image.fromarray((generated_image * 255).astype(np.uint8))
    generated_image_pil.save(generated_image_path)

    print("Original image saved at:", original_image_path)
    print("Generated image saved at:", generated_image_path)

    return jsonify({
        'original_image_path': original_image_path.replace(static_folder, "/static").replace("\\", "/"),  # Replace backslashes with forward slashes
        'generated_image_path': generated_image_path.replace(static_folder, "/static").replace("\\", "/")  # Replace backslashes with forward slashes
    })


# Hiển thị ảnh được tạo từ ảnh đầu vào
@app.route('/image_to_image_page')
def render_image_to_image_page():
    logged_in = session.get('logged_in', False)
    original_image_path = request.args.get('original_image_path')
    generated_image_path = request.args.get('generated_image_path')
    return render_template('image_to_image_page.html', 
                           original_image_path=original_image_path, 
                           generated_image_path=generated_image_path, 
                           logged_in=logged_in)






# Trang đăng nhập
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if check_user_info(username, password):
            session['logged_in'] = True
            session['username'] = username
            flash('Đăng nhập thành công!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Tên đăng nhập hoặc mật khẩu không đúng.', 'error')
    return render_template('login_page.html')




# Trang đăng ký
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        if username and password and email:
            add_new_user(username, password, email)
            flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Vui lòng điền đầy đủ thông tin.', 'error')
    return render_template('signup_page.html')



# @app.route('/search')
# def search():
#     logged_in = session.get('logged_in', False)
#     query = request.args.get('query')
#     conn = sqlite3.connect('imageweb.db')
#     cursor = conn.cursor()
#     cursor.execute("""
#         SELECT images.id, original_image_path, generated_image_path, title, collections.name AS collection_name
#         FROM images
#         LEFT JOIN collections ON images.collection_id = collections.id
#         WHERE title LIKE ? OR collections.name LIKE ?
#     """, ('%' + query + '%', '%' + query + '%'))
#     results = cursor.fetchall()
#     conn.close()
#     return render_template('search_results.html', query=query, results=results, logged_in=logged_in)


# # Tìm kiếm
@app.route('/search_results_page')
def search_results_page():
    logged_in = session.get('logged_in', False)
    query = request.args.get('query')
    conn = sqlite3.connect('imageweb.db')
    cursor = conn.cursor()

    # Thêm dấu `%` vào chuỗi tìm kiếm
    search_query = f"%{query}%"
    print(search_query)

    # Tìm kiếm trong bảng images
    cursor.execute("""SELECT generated_image_path, title 
                   FROM images 
                   WHERE title LIKE :search""", {"search": '%' + search_query + '%'})
    image_results = cursor.fetchall()
    print(image_results)

    # Tìm kiếm trong bảng collections
    cursor.execute("""
        SELECT i.generated_image_path, i.title
        FROM collections c 
        JOIN images i ON c.id = i.collection_id
        WHERE c.name LIKE :search""", {"search": '%' + search_query + '%'})
    collection_results = cursor.fetchall()

    cursor.execute("SELECT * from collections WHERE name LIKE ?", (search_query,))
    img_rs = cursor.fetchall()
    print(img_rs)

    combined_results = list(set(image_results + collection_results))
    print("Combined Results:", combined_results)

    return render_template('search_results_page.html', results=combined_results, logged_in=logged_in)


# test
# @app.route('/search_results_page')
# def search_results_page():
#     logged_in = session.get('logged_in', False)
#     query = request.args.get('query')
#     conn = sqlite3.connect('imageweb.db')
#     cursor = conn.cursor()

#     # Thêm dấu `%` vào chuỗi tìm kiếm
#     search_query = f"{query}"
#     print(f"Search Query: {search_query}")

#     # Tìm kiếm trong bảng images
#     cursor.execute("SELECT generated_image_path, title FROM images WHERE title = :search", {"search": '%' + search_query + '%'})
#     image_results = cursor.fetchall()
#     print(f"Image Results: {image_results}")


#     # Kiểm tra việc JOIN collections và images mà không có điều kiện LIKE
#     cursor.execute("""
#         SELECT i.generated_image_path, c.name as title 
#         FROM collections c 
#         JOIN images i ON c.id = i.collection_id
#     """)
#     join_check_results = cursor.fetchall()
#     print(f"Join Check Results: {join_check_results}")

#     # Tìm kiếm trong bảng collections
#     cursor.execute("""
#         SELECT i.generated_image_path, c.name as title 
#         FROM collections c 
#         JOIN images i ON c.id = i.collection_id
#         WHERE c.name LIKE :search""", {"search": '%' + search_query + '%'})
#     collection_results = cursor.fetchall()
#     print(f"Collection Results: {collection_results}")


#     cursor.execute("""
#         SELECT i.generated_image_path, c.name as title 
#         FROM collections c 
#         JOIN images i ON c.id = i.collection_id
#         WHERE c.name = 'test 1'""")
#     collection_results2 = cursor.fetchall()
#     print(f"Collection Results2: {collection_results2}")

#     results = image_results + collection_results
#     print(f"Combined Results: {results}")

#     conn.close()

#     return render_template('search_results_page.html', results=results, logged_in=logged_in)



# Thêm ảnh vào bộ sưu tập
@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.get_json()
    collection_name = data.get('collection_name')
    image_name = data.get('image_name')
    original_image_path = data.get('originalImagePath').replace(static_folder, "/static").replace("\\", "/")
    generated_image_path = data.get('generatedImagePath').replace(static_folder, "/static").replace("\\", "/")
    new_tag_id = data.get('tag_id')  # Lấy tag_id từ dữ liệu gửi lên từ front end
    username = session.get('username')

    if not username:
        return jsonify({'error': 'User not logged in'}), 403

    user_id = get_user_id(username)
    if not user_id:
        return jsonify({'error': 'User not found'}), 404

    collection_id = get_collection_id(collection_name, user_id)
    if not collection_id:
        add_collection(collection_name, user_id)
        collection_id = get_collection_id(collection_name, user_id)

    image_id = get_image_id_by_path(generated_image_path)
    if not image_id:
        image_id = add_image(original_image_path, generated_image_path, image_name)
    
    add_image_to_collection(image_id, collection_id)

    tag_id = get_tag_id_by_image(image_id)
    if not tag_id:
        add_tag_to_image(image_id, new_tag_id)

    return jsonify({'message': 'Image saved successfully'}), 200



# Trang chính
@app.route('/')
@app.route('/index')
def home():
    logged_in = session.get('logged_in', False)
    return render_template('index.html', logged_in=logged_in)


# Đăng xuất
@app.route('/logout')
def logout():
    session['logged_in'] = False
    session.pop('username', None)
    flash('Bạn đã đăng xuất thành công!', 'success')
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
