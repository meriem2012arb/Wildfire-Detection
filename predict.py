import json
import numpy as np
from PIL import Image
import requests
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from flask import Flask
from flask import request


app = Flask('fireforest')

@app.route('/predict', methods=['POST'])

def predict():
    
    
    model = load_model('model.tflite')

    inputImg = request.get_json()

    classes = {'fire': 0, 'nofire': 1}

    img = Image.open(requests.get(inputImg['im'], stream=True).raw)
    img = img.resize((150,150 ))

    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img /= 255
    res = model.predict(img)
    y_class = np.round(res)
    res = (res*100)
    y = list(classes)
    y_class = int(y_class)

    
    result = {
        'Probability NO FIRE:': round(float(res), 4),
        'Result': str(y[y_class])
            }


   

    print(result)
    return json.dumps(result)
    

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)




