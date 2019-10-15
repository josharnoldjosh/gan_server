#Define the function to convert a seg image to synthetic image
#Author Mingyang Zhou
import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from util.util import tensor2im
from data.base_dataset import get_params, get_transform
from PIL import Image
import numpy as np

from io import StringIO

# test
from PIL import Image
from io import BytesIO
import base64
from math import sqrt

class Seg2Real:

    def __init__(self):     
        self.opt = TestOptions().parse()                
        self.opt.crop_size = 512
        self.opt.load_size = 512
        self.opt.no_instance = True
        self.opt.preprocess_mode= "scale_width"
        self.opt.dataset_mode = "custom"
        self.opt.name = "landscape_pretrained"
        self.opt.cache_filelist_read = False
        self.opt.cache_filelist_write = False
        self.opt.semantic_nc = 182
        self.opt.contain_dontcare_label = False
        
        #Load the model
        self.model = Pix2PixModel(self.opt)
        self.model.eval()

        self.map = {
            110:156,
            60:154,
            128:134,
            99:149,
            108:126, #
            242:105,
            214:14,
            145:124,
            237:158,
            101:147, #
            70:96,
            135:168,
            65:148,
            209:143,
            150:110,
            83:125,
            120:135,
            141:119,
            156:161,
            82:177,
            230:118,
            113:123,
            232:162
        }
        return

    def preprocess_segmap(self, seg_img):
        converting_map = {
            110:156,
            60:154,
            128:134,
            99:149,
            108:126, #
            242:105,
            214:14,
            145:124,
            237:158,
            101:147, #
            70:96,
            135:168,
            65:148,
            209:143,
            150:110,
            83:125,
            120:135,
            141:119,
            156:161,
            82:177,
            230:118,
            113:123,
            232:162
        }
        for i in range(seg_img.shape[0]):
            for j in range(seg_img.shape[1]):
                if seg_img[i][j] == 0:
                    seg_img[i][j] = 110
                        
                if seg_img[i][j] in converting_map.keys():
                    seg_img[i][j] = converting_map[seg_img[i][j]]
    #             else:                   
    #                 seg_img[i][j] = 156
        
        return seg_img

    def closest_dominant(self, sample_matrix, d_class, x,y):
        #First extract all the (x,y) positions of a certain label
        min_distance = 10000
        min_class = None
        for sample_class in d_class:
            sample_class_indexes = np.where(sample_matrix==sample_class)
            points = [(x,y) for x,y in zip(sample_class_indexes[0],sample_class_indexes[1])]
            points.sort(key = lambda p: sqrt((p[0] - x)**2 + (p[1] - y)**2))
            current_distance = sqrt((points[0][0] - x)**2 + (points[0][1] - y)**2)
            if current_distance < min_distance:
                min_class = sample_class
                min_distance = current_distance
        return min_class

    def merge_noisy_pixels(self, sample_matrix, d_class,kernel_width=3, kernel_height=3,sliding_size=1):
        #First extract all the (x,y) positions of a certain label
        i = 0
        j = 0
        converted_pixels = 0
        while i + kernel_width <= sample_matrix.shape[1]:
            while j + kernel_height <= sample_matrix.shape[0]:
                current_window = sample_matrix[j:j+kernel_width,i:i+kernel_height]
                current_window_list = current_window.flatten().tolist()
                # Check whether there is unknown labels in current_window
                current_labels = set(current_window_list)
                if not current_labels.issubset(set(d_class)):
                    #replace the noisy labels with the closes 
                    d_class_subset = list(current_labels.intersection(set(d_class)))
                    # replace the noisy labels with the dominant class 
                    d_class_subset.sort(key = lambda p : current_window_list.count(p), reverse=True)
                    dominant_d_class = d_class_subset[0]
                    sample_matrix[j:j+kernel_width][i:i+kernel_height] = dominant_d_class
                    mask = ~ np.isin(current_window, d_class)
                    converted_pixels += np.sum(mask)
                    current_window[mask] = dominant_d_class
                    sample_matrix[j:j+kernel_width,i:i+kernel_height] = current_window
                j += sliding_size
            i += sliding_size
            j = 0
        return sample_matrix
       
    def smooth_image(self, raw_img, gt_img, dominant_thred=2000):
        """
        The new smooth function for semantic image from drawer
        """
        #Preprocess with segmentation map
        raw_img = self.preprocess_segmap(raw_img)
        
        #Define the landscape labels
        landscape_labels = [110, 124, 125, 135, 139, 148, 14, 105, 119, 126, 134, 147, 149, 154, 156, 158, 161, 177, 96, 118, 123, 162, 168, 181]
        
        #Find dominant labels in the image
        a,b= np.histogram(gt_img,bins=np.unique(gt_img))
        result = np.where(a > dominant_thred)
        dominant_class = []
        for x in range(result[0].shape[0]):
            if b[result[0][x]] in landscape_labels:
                dominant_class.append(b[result[0][x]])
        
        result_img = self.merge_noisy_pixels(raw_img,dominant_class)
        
        return result_img

    def unionCategory_RelevantPos(self, label_gt, label_pred, n_class):
        hist = np.zeros((n_class, n_class))
        #loop through the matrix row by row
        for lt, lp in zip(label_gt, label_pred):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        
        #print(hist)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        
        #print(iu)
        
    #     #loop through the list of items and identify the categories
    #     intersect_labels = np.where(iu > 0)
    #     union_labels = np.where(~np.isnan(iu))
    #     for l in np.nditer(intersect_labels):
    #         gt_mask = np.zeros(label_gt.shape,dtype=int)
    #         pred_mask = np.zeros(label_pred.shape,dtype=int)
    #         #Get the masks of the label in gt_matrix
    #         gt_label_map = label_gt == l
    #         gt_mask[gt_label_map]=1
    #         #Get the masks of the label in pred_matrix
    #         pred_label_map = 
    #         break


    def seg2real(self, ground_truth_image, seg_img):
        """
        Given a saved semantic segmentation labeling image path, this function
        will load that semantic segmentation image and generate a synthetic real image
        from the pretrained land scape model. The image will then be saved at target_path
        Input:
          image_path: path where you save your semantic label image
          target_path: path where you will save your synthetic real image
          self.opt: The configuration dict for models
        """

        # resize image
        ground_truth_image = ground_truth_image.resize((350, 350))
        seg_img = seg_img.resize((350, 350))

        # convert to array  
        seg_img = np.array(seg_img)
        ground_truth_image = np.array(ground_truth_image)
        
        # gonna have to update this badboy
        for i in range(seg_img.shape[0]):
            for j in range(seg_img.shape[1]):
                if seg_img[i][j] == 0:
                    seg_img[i][j] = 110
                    
                if seg_img[i][j] in self.map.keys():
                    seg_img[i][j] = self.map[seg_img[i][j]]
                else:                   
                    seg_img[i][j] = 156

        # seg_img = self.smooth_image(seg_img, ground_truth_image)
        # print(seg_img)

        print("ground_truth_image", ground_truth_image)
        print("——————")
        print("seg_img", seg_img) 

        # calculate metrics
        scores = self.calculate_metrics(ground_truth_image, seg_img)

        seg_img = Image.fromarray(seg_img)

        #Get data item
        # seg_img = Image.open(image_path)
        params = get_params(self.opt, seg_img.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        seg_tensor = transform_label(seg_img) * 255.0
        seg_tensor[seg_tensor == 255] = self.opt.label_nc 
        seg_tensor = seg_tensor.unsqueeze(0)
        #seg_img =      

        model_input = {"label": seg_tensor,
                       "instance": seg_tensor,
                       "image":seg_tensor}

        #Process the label
        generated = self.model(model_input, mode='inference')

        #convert tensor to image
        generated_img = tensor2im(generated)
        
        synthetic_img = Image.fromarray(generated_img[0])

        return (synthetic_img, scores, seg_img)

    def calculate_metrics(self, ground_truth_image, drawer_image):
        # return {"pixel_acc":0, "mean_acc":0, "mean_iou":0, "mean_iou_class":0}
        print(ground_truth_image.shape, drawer_image.shape)
        if ground_truth_image[0][0] == 1 or ground_truth_image.shape != drawer_image.shape:
            print("SHAPES NOT EQUAL OR IMAGE DOES NOT EXIST")
            return {"pixel_acc":0, "mean_acc":0, "mean_iou":0}
        
        # np.savetxt('test1.txt', ground_truth_image, fmt='%d')
        # np.savetxt('test2.txt', drawer_image, fmt='%d')

        # print(ground_truth_image.shape, drawer_image.shape)
        # print(ground_truth_image, drawer_image)
        # print(ground_truth_image.dtype, drawer_image.dtype)

        pixel_accuracy = self.pixel_accuracy(ground_truth_image, drawer_image, 182)     
        mean_accuracy = self.mean_accuracy(ground_truth_image, drawer_image, 182)
        mean_IoU = self.mean_IoU(ground_truth_image, drawer_image, 182)     
        class_IoU = self.class_IoU(ground_truth_image, drawer_image, 182)

        return {"pixel_acc":pixel_accuracy, "mean_acc":mean_accuracy, "mean_iou":mean_IoU}

        #Fork the function from deeplabv2
    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        #print(hist)
        return hist

    def pixel_accuracy(self, label_gt, label_pred, n_class):
        hist = np.zeros((n_class, n_class))
        #loop throught the matrix row by row
        for lt, lp in zip(label_gt, label_pred):
            hist += self._fast_hist(lt.flatten(), lp.flatten(), n_class)
        #The [i,j] the element in hist indicate the number of values when gt_matrix = i and pred_matrix =j 
        acc = np.diag(hist).sum() / hist.sum()
        return acc
    def mean_accuracy(self, label_gt, label_pred, n_class):
        hist = np.zeros((n_class, n_class))
        #loop throught the matrix row by row
        for lt, lp in zip(label_gt, label_pred):
            hist += self._fast_hist(lt.flatten(), lp.flatten(), n_class)
        #The [i,j] the element in hist indicate the number of values when gt_matrix = i and pred_matrix =j 
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        return acc_cls
    #Mean_IOU is the metrics that we want to use
    def mean_IoU(self, label_gt, label_pred, n_class):
        hist = np.zeros((n_class, n_class))
        #loop throught the matrix row by row
        for lt, lp in zip(label_gt, label_pred):
            hist += self._fast_hist(lt.flatten(), lp.flatten(), n_class)
        
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        #print(iu)
        valid = hist.sum(axis=1) > 0  # added
        #print(valid)
        mean_iu = np.nanmean(iu[valid])
        
        return mean_iu

    def class_IoU(self, label_gt, label_pred, n_class):
        hist = np.zeros((n_class, n_class))
        #loop throught the matrix row by row
        for lt, lp in zip(label_gt, label_pred):
            hist += self._fast_hist(lt.flatten(), lp.flatten(), n_class)
        
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        cls_iu = dict(zip(range(n_class), iu))
        
        return cls_iu

if __name__ == '__main__':    
    ground_truth = Image.open('./test1.png').convert('L')
    if ground_truth.size[0] < ground_truth.size[1]:
        ground_truth = ground_truth.crop((0, 0, ground_truth.size[0], ground_truth.size[0])).resize((348, 350))
    else:
        ground_truth = ground_truth.crop((0, 0, ground_truth.size[1], ground_truth.size[1])).resize((348, 350))  
    (synthetic_img, scores) = Seg2Real().seg2real(ground_truth, Image.open('./test2.jpg'))
    print(scores)