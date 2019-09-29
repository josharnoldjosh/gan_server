from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy
from io import StringIO

# test
from PIL import Image
from io import BytesIO
import base64

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

from seg2real import Seg2Real

import time

# Init the server
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)

model = Seg2Real()

@app.route('/get_image', methods = ['GET', 'POST', 'PATCH', 'PUT', 'OPTIONS'])
def get_image():    
    image_name = request.form['image_name'] # 8935320126_64a018d425_o.jpg    
    print('test', image_name)
    image = Image.open('landscape_target/'+image_name)
    buffered = BytesIO()
    image.save(buffered, format="png")        
    img_str = 'data:image/png;base64,'+base64.b64encode(buffered.getvalue()).decode('ascii')
    return img_str

@app.route('/peek', methods = ['GET', 'POST', 'PATCH', 'PUT', 'OPTIONS'])
def peek():
    if request.method == 'POST':
        unique_id = request.form['unique_id']
        turn_idx = request.form['turn_idx']        
        path_file = "./saved_data/"+unique_id+"_"+turn_idx+"_synthetic.jpg"
        image = Image.open(path_file)
        buffered = BytesIO()
        image.save(buffered, format="png")                
        img_str = 'data:image/png;base64,'+base64.b64encode(buffered.getvalue()).decode('ascii')
        return img_str
    return ""

# @app.route('/save_image', methods = ['GET', 'POST', 'PATCH', 'PUT', 'OPTIONS'])
# def save_image():
#     if request.method == 'POST':

#         data = request.form['file']
#         py_data = data.replace('data:image/png;base64,','')
#         image = Image.open(BytesIO(base64.b64decode(py_data))).convert('RGB') # Drop alpha

#         unique_id = request.form['unique_id']
#         turn_idx = request.form['turn_idx']
#         image_name = request.form['image_name'] # 8935320126_64a018d425_o.jpg    
#         print('test', image_name)
#         image = Image.open('landscape_target/'+image_name)

#         image.save("./saved_data/"+unique_id+"_"+turn_idx+"_real.jpg")

#         (red, green, blue) = image.split()        
#         image = red.convert('L')
#         image = model.seg2real(image, image)    
#         buffered = BytesIO()
#         image.save(buffered, format="png")        
#         image.save("./saved_data/"+unique_id+"_"+turn_idx+"_synthetic.jpg")
#     return ""

@app.route('/', methods = ['GET', 'POST', 'PATCH', 'PUT', 'OPTIONS'])
def root():
    if request.method == 'POST':              

        # load data
        data = request.form['file']
        unique_id = request.form['unique_id']
        turn_idx = request.form['turn_idx']
        image_name = request.form['image_name'] # 8935320126_64a018d425_o.jpg    

        # save real
        py_data = data.replace('data:image/png;base64,','')
        image = Image.open(BytesIO(base64.b64decode(py_data))).convert('RGB') # Drop alpha
        (red, green, blue) = image.split()        
        image = red.convert('L')
        image.save("./saved_data/"+unique_id+"_"+turn_idx+"_real.jpg")

        # get ground truth
        ground_truth = numpy.ones((350,348),dtype=int)
        if image_name != "undefined":
            ground_truth = Image.open('landscape_target/'+image_name).convert('L')

        # pass through seg2real
        image = model.seg2real(ground_truth, image)
        
        # save synthetic & return
        buffered = BytesIO()
        image.save(buffered, format="png") 
        image.save("./saved_data/"+unique_id+"_"+turn_idx+"_synthetic.jpg")           
        img_str = 'data:image/png;base64,'+base64.b64encode(buffered.getvalue()).decode('ascii')    
        return img_str
    else:
        return "get request"

if __name__ == '__main__':
    app.run(port=1234)