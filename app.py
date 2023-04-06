from flask import Flask, request, jsonify
import keras
import cv2
import numpy as np
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
labels=["Apple","Avocado","Banana","Beach Plum"," Blueberry","Cashewnut","Coconut","Custard Apple","Date","Dragonfruit","Finger Lime","Gooseberry","Grape","Guava","Horned Melon","Jackfruit","Kiwi","Lemon","Lime","Mango","Muskmelon","Orange","Papaya","Peanut","Persimmon","Pomelo","Raspberry","Strawberry"]
model = keras.models.load_model('BestModel_v3.h5')

# Define the endpoint for image classification
@app.route('/classify', methods=['POST'])
def classify():
    img_file = request.files['image']
    img_bytes = np.asarray(bytearray(img_file.read()), dtype="uint8")   
    img = cv2.imdecode(img_bytes, cv2.COLOR_BGR2RGB)  
    resized_img = cv2.resize(img, (150, 150))   
    normalized_img = resized_img / 255.0    

    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)  
    data[0] = normalized_img

    predictions = model.predict(data)

    # # Return the result as JSON  
    # index = np.argmax(prediction[0]) 
    
    # return jsonify({ 'class': labels[index] })

    top_5 = np.argsort(-predictions[0])[:5]  # Descending order
    
    results = []
    for i in top_5:
        label = labels[i]
        confidence = round(predictions[0][i]*100, 2)  
        results.append({'class': label, 'confidence': str(confidence)+'%'})
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')