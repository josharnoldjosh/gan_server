from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from math import sqrt
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

import cv2
import matplotlib.pyplot as plt
from skimage import io, morphology, measure
from scipy import ndimage
from scipy.ndimage import center_of_mass
from scipy import ndimage

# Init the server
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)

model = Seg2Real()

def best_smooth_method(image, dominant_thred=1000):
    r,g,b = cv2.split(image)
    unique_class, unique_counts = np.unique(r, return_counts=True)
    result = np.where(unique_counts > dominant_thred)
    dominant_class = unique_class[result].tolist()
    num_cluster = len(dominant_class)
    #Reshape the image
    image_2D = image.reshape((image.shape[0]*image.shape[1],3))
    # convert to np.float32
    image_2D = np.float32(image_2D)
    #define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center = cv2.kmeans(image_2D, num_cluster, None,criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    #Mapping the segmented image to labels
    drawing2landscape = [
        ([0, 0, 0],156), #sky
        ([110, 180, 232], 156),#sky
        ([60, 83, 163], 154), #sea
        ([128, 119, 97], 134), #mountain
        ([99, 95, 93], 149), #rock
        ([108, 148, 96], 126), #hill
        ([242, 218, 227], 105), #clouds
        ([214, 199, 124], 14), #sand
        ([145, 132, 145], 124), #gravel
        ([237, 237, 237], 158), #snow
        ([101, 163, 152], 147), #river
        ([70, 150, 50], 96), #bush
        ([135, 171, 111], 168), #tree
        ([65, 74, 74], 148), #road
        ([150, 126, 84], 110), #dirt 
        ([120, 75, 38], 135), #mud 
        ([141, 159, 184], 119), #fog 
        ([156, 156, 156], 161), #stone
        ([82, 107, 217], 177), #water
        ([230, 190, 48], 118), #flower
        ([113, 204, 43], 123), #grass
        ([232, 212, 35], 162), #straw
    ]

    #Find the closest labels
    label = label.reshape((image.shape[:2]))
    #map label to corresponding tag
    converstion = {}
    for i, c in enumerate(center):
        #sort the defined centers
        drawing2landscape.sort(key = lambda p: sqrt((p[0][0] - c[0])**2 + (p[0][1] - c[1])**2 + (p[0][2] - c[2])**2))
        #construct the mapping
        label[np.where(label==i)] = drawing2landscape[0][1]
    return label

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
        turn_idx = int(request.form['turn_idx'])        
        path_file = "./saved_data/"+unique_id+"_"+str(turn_idx)+"_synthetic.jpg"
        image = None

        while not os.path.exists(path_file) and turn_idx >= 0:            
            path_file = "./saved_data/"+unique_id+"_"+str(turn_idx)+"_synthetic.jpg"
            turn_idx -= 1

        print(path_file)

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
        turn_idx = int(request.form['turn_idx'])

        # try to find the last score file that exists, go into negative score files
        while not os.path.exists("./saved_data/"+unique_id+"_"+str(turn_idx)+'_score'+'.json') and turn_idx > -7:
            turn_idx -= 1

        # if our score file is positive we know its legit
        if turn_idx  >= 0:
            with open("./saved_data/"+unique_id+"_"+str(turn_idx)+'_score'+'.json', 'r') as f:
                print(f)
                result = json.load(f)
                print(result)
                return result
    return {"pixel_acc":0, "mean_acc":0, "mean_iou":0, "co_draw":0}

@app.route('/sandbox', methods = ['GET', 'POST', 'PATCH', 'PUT', 'OPTIONS'])
def sandbox():
    if request.method == 'POST':              

        # load data
        data = request.form['file']
        unique_id = request.form['unique_id']
        turn_idx = request.form['turn_idx']
        image_name = request.form['image_name'] # 8935320126_64a018d425_o.jpg  
        py_data = data.replace('data:image/png;base64,','')

        # Semantic label image directly from canvas
        image = Image.open(BytesIO(base64.b64decode(py_data)))
        image = image.convert('RGB').resize((350, 350))          

        # test smooth
        test = np.array(image)
        smoothed_image = best_smooth_method(test) # should give us an array
        test_smooth_output = Image.fromarray(np.uint8(smoothed_image))

        # test the assert ==
 
        # pass through seg2real
        (image, scores, seg_img) = model.seg2real(smoothed_image, smoothed_image, False)

        # save real        
        seg_img.resize((350, 350))
        
        image.save("./saved_data/"+unique_id+"_"+turn_idx+"_synthetic.jpg")
        
        # save synthetic & return
        buffered = BytesIO()
        image.save(buffered, format="png") 
        img_str = 'data:image/png;base64,'+base64.b64encode(buffered.getvalue()).decode('ascii')    
        return img_str
    else:
        return "get request"

@app.route('/', methods = ['GET', 'POST', 'PATCH', 'PUT', 'OPTIONS'])
def root():
    if request.method == 'POST':              

        # load data
        data = request.form['file']
        unique_id = request.form['unique_id']
        turn_idx = request.form['turn_idx']
        image_name = request.form['image_name'] # 8935320126_64a018d425_o.jpg  
        py_data = data.replace('data:image/png;base64,','')

        # Semantic label image directly from canvas
        image = Image.open(BytesIO(base64.b64decode(py_data)))
        image = image.convert('RGB').resize((350, 350))  
        image.save("./saved_data/"+unique_id+"_"+turn_idx+"_raw_label.png")

        # test smooth
        test = np.array(image)
        smoothed_image = best_smooth_method(test) # should give us an array
        test_smooth_output = Image.fromarray(np.uint8(smoothed_image))
        # test_smooth_output.save("./saved_data/"+unique_id+"_"+turn_idx+"_raw_label.png")
        
        # continue
        # image.resize((350, 350)).save("./saved_data/"+unique_id+"_"+turn_idx+"_raw_label.jpg")
        # (red, green, blue) = image.split()        
        # image = red.convert('L')

        # get ground truth
        ground_truth = np.ones((350,350),dtype=int)
        if image_name != "undefined":
            ground_truth = Image.open('landscape_target/'+image_name.replace('.jpg', '_semantic.png')).convert('L')
            if ground_truth.size[0] < ground_truth.size[1]:
                ground_truth = ground_truth.crop((0, 0, ground_truth.size[0], ground_truth.size[0])).resize((350, 350))
            else:
                ground_truth = ground_truth.crop((0, 0, ground_truth.size[1], ground_truth.size[1])).resize((350, 350))  

        # pass through seg2real
        (image, scores, seg_img) = model.seg2real(ground_truth, smoothed_image, True)

        # save real        
        seg_img.resize((350, 350)).save("./saved_data/"+unique_id+"_"+turn_idx+"_real.jpg")

        print(scores)

        # save scores        
        scores_file_path = "./saved_data/"+unique_id+"_"+turn_idx+'_score'+'.json'        
        with open(scores_file_path, 'w') as f:
            json.dump(scores, f)
            print("saved to", scores_file_path)

        # image.save("./saved_data/"+unique_id+"_"+turn_idx+"_synthetic.jpg")

        # save synthetic & return
        buffered = BytesIO()
        image.save(buffered, format="png")                 
        img_str = 'data:image/png;base64,'+base64.b64encode(buffered.getvalue()).decode('ascii')    
        return img_str
    else:
        return "get request"

@app.route('/test', methods = ['GET', 'POST', 'PATCH', 'PUT', 'OPTIONS'])
def test():           
    
    # Semantic label image directly from canvas
    image = Image.open('2.png')
    image = image.convert('RGB').resize((350, 350))      

    # Smooth
    test = np.array(image)
    smoothed_image = best_smooth_method(test)
    test_smooth_output = Image.fromarray(np.uint8(smoothed_image))    
    

    # get ground truth
    ground_truth = Image.open('2_target.png').convert('L').resize((350, 350))

    # pass through seg2real
    (image, scores, seg_img) = model.seg2real(ground_truth, smoothed_image, True)

    return scores

if __name__ == '__main__':
    app.run(port=1234)