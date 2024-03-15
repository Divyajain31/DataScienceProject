from flask import Flask,render_template,request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# define the flask app
app=Flask(__name__)

# load the model
model=load_model(r"C:\Users\aksha\OneDrive\Desktop\COLLEGE\LW_Summer\plant proj\plant proj\models\plant-224.h5")

def model_predict(img_path,model):
    test_image=image.load_img(img_path,target_size=(224,224))
    test_image=image.img_to_array(test_image)
    test_image=test_image/255
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    return result


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        # get the file from post request
        f=request.files['file']

        # save the file to uploads folder
        basepath=os.path.dirname(os.path.realpath('__file__'))
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result =model_predict(file_path,model)

        categories= ['Apple___Apple_scab',
              'Apple___Black_rot',
              'Apple___Cedar_apple_rust',
              'Apple___healthy',
              'Blueberry___healthy',
              'Cherry_(including_sour)___healthy',
              'Cherry_(including_sour)___Powdery_mildew',
              'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
              'Corn_(maize)___Common_rust_','Corn_(maize)___healthy',
              'Corn_(maize)___Northern_Leaf_Blight',
              'Grape___Black_rot',
              'Grape___Esca_(Black_Measles)',
              'Grape___healthy',
              'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
              'Orange___Haunglongbing_(Citrus_greening)',
              'Peach___Bacterial_spot','Peach___healthy',
              'Pepper,_bell___Bacterial_spot',
              'Pepper,_bell___healthy',
              'Potato___Early_blight',
              'Potato___healthy',
              'Potato___Late_blight',
              'Raspberry___healthy',
              'Soybean___healthy',
              'Squash___Powdery_mildew',
              'Strawberry___healthy',
              'Strawberry___Leaf_scorch',
              'Tomato___Bacterial_spot',
              'Tomato___Early_blight',
              'Tomato___healthy',
              'Tomato___Late_blight',
              'Tomato___Leaf_Mold',
              'Tomato___Septoria_leaf_spot',
              'Tomato___Spider_mites Two-spotted_spider_mite',
              'Tomato___Target_Spot',
              'Tomato___Tomato_mosaic_virus',
              'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

        # process your result for human
        pred_class = result.argmax()
        output=categories[pred_class]
        return output
    return None

if __name__=='__main__':
    app.run(debug=False,port=5926)
