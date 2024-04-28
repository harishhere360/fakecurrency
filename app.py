import numpy as np
from flask import Flask, request, render_template, redirect, url_for

import cv2
import tensorflow as tf
from keras.models import load_model
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dummy user credentials
USER_CREDENTIALS = {
    'username': 'admin',
    'password': 'admin'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_jpg_image(img):
    img = tf.convert_to_tensor(img[:,:,:3])
    img = np.expand_dims(img, axis=0)
    img = tf.image.resize(img,[224,224])
    img = (img/255.0)
    return img

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username == USER_CREDENTIALS['username'] and password == USER_CREDENTIALS['password']:
        return redirect(url_for('home'))
    else:
        return render_template('login.html', error_message='Invalid credentials. Please try again.')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text='No file uploaded!')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', prediction_text='No selected file!')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        model = load_model('vgg.h5')  # Load the model here to reduce memory usage
        
        class_names = [('fake', 0), ('real', 1)]
        test_image_read_1 = cv2.imread(file_path)
        
        if test_image_read_1 is None:
            return render_template('index.html', prediction_text='Failed to read the image file!')
        
        test_image_1 = process_jpg_image(test_image_read_1)
        prediction_1 = model.predict(test_image_1)
        
        prediction = int(np.argmax(prediction_1))
        currency_type = class_names[prediction][0]
        
        return render_template('index.html', prediction_text=f'The currency is {currency_type}')

if __name__ == "__main__":
    app.run(debug=True)
