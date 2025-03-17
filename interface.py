from flask import Flask, render_template, request, jsonify, url_for
import os
from datetime import datetime
from scripts.processing import process_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    filename = f"processed_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    processed_image = process_image(file)
    processed_image.save(filepath, format='PNG')

    return jsonify({'image_url': url_for('static', filename=f'uploads/{filename}', _external=True)})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
