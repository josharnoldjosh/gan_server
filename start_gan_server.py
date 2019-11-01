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
os.environ["CUDA_VISIBLE_DEVICES"]="5"

from seg2real import Seg2Real

import time
import json

import cv2
import matplotlib.pyplot as plt
from skimage import io, morphology, measure
from scipy import ndimage
from scipy.ndimage import center_of_mass
from scipy import ndimage

import enchant
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import re

# Init the server
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)

model = Seg2Real()

punctuation_string = string.punctuation
single_letter_word = "ai"

eng_dict = enchant.Dict("en_US")
eng_contractions = {
    "ain't": "am not / are not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is",
    "i'd": "I had / I would",
    "i'd've": "I would have",
    "i'll": "I shall / I will",
    "i'll've": "I shall have / I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}

def preprocess_utterance(utterance):
    utterance = utterance.lower()
    for key in eng_contractions.keys():
        if re.search(key, utterance):
            utterance = re.sub(key, "", utterance)
    return utterance


def detect_non_englishwords(utterance):
    """
    Given a english sentence, this function will give a list of non-english word in the utterance
    """
    # Step 1 Preprocess the Sentence to remove all the contraction phrase
    processed_utterance = preprocess_utterance(utterance)

    # Step 2 Tokenize the Processed sentence into list of words
    word_list = word_tokenize(processed_utterance)

    # Step 3 Detect Non-English Words and return i
    non_english = []
    for w in word_list:
        if (not eng_dict.check(w) or (w not in single_letter_word and len(w) == 1)) and w not in punctuation_string:
            non_english.append(w)
    return non_english

def best_smooth_method(image,dominant_thred=1000):
    r,g,b = cv2.split(image)

    #Find dominant labels in the image
    unique_class, unique_counts = np.unique(r, return_counts=True)
    #Need to smooth gt_labels as well
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
        
        # save synthetic & return
        buffered = BytesIO()
        image.save(buffered, format="png") 
        image.save("./saved_data/"+unique_id+"_"+turn_idx+"_synthetic.jpg")           
        img_str = 'data:image/png;base64,'+base64.b64encode(buffered.getvalue()).decode('ascii')    
        return img_str
    else:
        return "get request"

@app.route('/test', methods = ['GET', 'POST', 'PATCH', 'PUT', 'OPTIONS'])
def test():            
    # Semantic label image directly from canvas
    image = Image.open('3.png')
    image = image.convert('RGB').resize((350, 350))  

    test = np.array(image)
    smoothed_image = best_smooth_method(test) # should give us an array
    test_smooth_output = Image.fromarray(np.uint8(smoothed_image))

    ground_truth = Image.open('3_target.png').convert('L')
    
    # pass through seg2real
    (image, scores, seg_img) = model.seg2real(ground_truth, smoothed_image, True)  
    return scores

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

@app.route('/english', methods = ['GET', 'POST', 'PATCH', 'PUT', 'OPTIONS'])
def english():
    if request.method == 'POST':                        
        utt = request.form['utt']
        print(utt)
        try:
            utt = preprocess_utterance(utt)
            result = detect_non_englishwords(utt)
            info = "Sorry! Please only use proper english words when communicating. The following words you used are not english words: " + ",".join(result)
            can_send = len(result) == 0
            return {'info':info, 'can_send':can_send}
        except Exception as error:
            print("ERROR", error)
            return {'data':[]}    
    return 'test'
    
if __name__ == '__main__':
    app.run(port=1234)