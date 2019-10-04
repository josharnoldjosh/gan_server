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
import json

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
    if image.size[0] < image.size[1]:
        image = image.crop((0, 0, image.size[0], image.size[0]))
    else:
        image = image.crop((0, 0, image.size[1], image.size[1]))

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
        image = None

        while not os.path.exists(path_file) and turn_idx > 0:            
            path_file = "./saved_data/"+unique_id+"_"+turn_idx+"_synthetic.jpg"
            turn_idx -= 1

        if os.path.exists(path_file):
            image = Image.open(path_file)
        else:
            image = Image.new('L', (348, 350)) 

        buffered = BytesIO()
        image.save(buffered, format="png")                
        img_str = 'data:image/png;base64,'+base64.b64encode(buffered.getvalue()).decode('ascii')
        return img_str                
    return ""

@app.route('/get_score', methods = ['GET', 'POST', 'PATCH', 'PUT', 'OPTIONS'])
def get_score():
    if request.method == 'POST':
        unique_id = request.form['unique_id']
        turn_idx = request.form['turn_idx']
        print(unique_id, turn_idx, "this is a test")
        with open("./saved_data/"+unique_id+"_"+str(int(turn_idx)-1)+'_score'+'.json', 'r') as f:
            print(f)
            result = json.load(f)
            print(result)
            return result
    return {"pixel_acc":0, "mean_acc":0, "mean_iou":0, "mean_iou_class":0}

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
        ground_truth = numpy.ones((350,350),dtype=int)
        if image_name != "undefined":
            ground_truth = Image.open('landscape_target/'+image_name.replace('.jpg', '_semantic.png')).convert('L')
            if ground_truth.size[0] < ground_truth.size[1]:
                ground_truth = ground_truth.crop((0, 0, ground_truth.size[0], ground_truth.size[0])).resize((350, 350))
            else:
                ground_truth = ground_truth.crop((0, 0, ground_truth.size[1], ground_truth.size[1])).resize((350, 350))  

        # pass through seg2real
        (image, scores) = model.seg2real(ground_truth, image)

        print(scores)

        # save scores        
        scores_file_path = "./saved_data/"+unique_id+"_"+turn_idx+'_score'+'.json'        
        with open(scores_file_path, 'w') as f:
            json.dump(scores, f)
            print("saved to", scores_file_path)
        
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