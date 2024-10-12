from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

ecgModel = load_model('ecg_model.h5')

classes = ["HB", "MI", "N", "PMI"]

image_height = 64
image_width = 64


@app.route('/ecgPredict', methods=['POST'])
def ecgPredict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        img_bytes = file.read()
        img = image.img_to_array(image.load_img(io.BytesIO(img_bytes), target_size=(image_height, image_width)))
        img = np.expand_dims(img, axis=0)
        img /= 255.0 

        prediction = ecgModel.predict(img) 
        predicted_class_index = np.argmax(prediction)
        predicted_class = classes[predicted_class_index]
        predicted_probability = float(prediction[0][predicted_class_index]) * 100
        
        # Define messages for each class
        if predicted_probability >= 80:
                if predicted_class == 'MI':
                    message = f"Your ECG corresponds to Myocardial Infarction and percentage is : {predicted_probability:.2f}%"
                elif predicted_class == 'HB':
                    message = f"Your ECG corresponds to Abnormal Heartbeat and percentage is : {predicted_probability:.2f}%"
                elif predicted_class == 'N':
                    message = f"Your ECG is Normal and percentage is : {predicted_probability:.2f}%"
                elif predicted_class == 'PMI':
                    message = f"Your ECG corresponds to History of Myocardial Infarction and percentage is : {predicted_probability:.2f}%"
        else:
                message = f"This image is too low to make a reliable diagnosis. Try again"
        return jsonify({'prediction': message})



# if __name__ == '__main__':
#     app.run(host="192.168.8.162",port=5000)
if __name__ == '__main__':
    app.run()