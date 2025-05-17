from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import sys
import base64
import tempfile

# Add the parent directory to Python path to import predict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import predict_image

app = Flask(__name__)

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join('templates', path)):
        return send_from_directory('templates', path)
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a PNG or JPEG image.'}), 400

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            
            # Get prediction
            prediction = predict_image(temp_file.name)

            # Read the image file and convert to base64
            with open(temp_file.name, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')

            # Clean up
            os.unlink(temp_file.name)

            return jsonify({
                'success': True,
                'prediction': prediction,
                'image': img_data
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For local development
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 3000))) 