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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

from skimage import io, morphology, measure
from scipy import ndimage
from itertools import permutations
import cv2

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
        
    def seg2real(self, ground_truth_image, seg_img):
        # resize image
        ground_truth_image = ground_truth_image.resize((350, 350))
        # seg_img = seg_img.resize((350, 350))

        # convert to array  
        # seg_img = np.array(seg_img)
        ground_truth_image = np.array(ground_truth_image)
        
        # currently there is no smoothing
        # gonna have to update this badboy
        # for i in range(seg_img.shape[0]):
        #     for j in range(seg_img.shape[1]):
        #         if seg_img[i][j] == 0:
        #             seg_img[i][j] = 110
                    
        #         if seg_img[i][j] in self.map.keys():
        #             seg_img[i][j] = self.map[seg_img[i][j]]
        #         else:                   
        #             seg_img[i][j] = 156

        print("ground_truth_image", ground_truth_image)
        print("——————")
        print("seg_img", seg_img) 

        # calculate metrics
        scores = self.calculate_metrics(ground_truth_image, seg_img)

        seg_img = Image.fromarray(np.uint8(seg_img))

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
        # class_IoU = self.class_IoU(ground_truth_image, drawer_image, 182)

        co_draw_metric = self.gaugancodraw_eval_metrics(drawer_image, ground_truth_image, 182)

        print("TEST", co_draw_metric)

        return {"pixel_acc":pixel_accuracy, "mean_acc":mean_accuracy, "mean_iou":mean_IoU, "co_draw":co_draw_metric}
        
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

    def find_label_centers(self, image, shared_label, noisy_filter=1000):
        """
          find the centers of the labels in the image
        """
        image_shared = {l:None for l in shared_label}
        #construct the center for draw_shared
        for key in image_shared.keys():
            mask = np.int_((image == key))
            lbl = ndimage.label(mask)[0]
            unique_labels, unique_counts = np.unique(lbl,return_counts=True)
            filtered_labels = unique_labels[np.where(unique_counts > noisy_filter)]
            filtered_labels = filtered_labels[np.where(filtered_labels > 0)]
            
            centers = ndimage.measurements.center_of_mass(mask, lbl, filtered_labels)
            image_shared[key] = centers
        return image_shared

    def relevant_score(self, x,y):
        """
        x and y are two turples of the objects in drawed image and ground truth image.
        """
        
        score_x =  1 if (x[0][0]-y[0][0])*(x[1][0]-y[1][0]) > 0 else 0 
        score_y =  1 if (x[0][1]-y[0][1])*(x[1][1]-y[1][1]) > 0 else 0
        
        return score_x+score_y

    def relevant_eval_metrics(self, draw_smooth, gt_smooth, g_smooth_thred=1000):
        """
        Compute the relevant_eval_metrics from draw_segments and ground_truth_segments
        """ 

        #Now let's compute the relevant locations
        draw_set = np.unique(draw_smooth).tolist()
        print(draw_set)
        gt_set = np.unique(gt_smooth).tolist()
        print(gt_set)
        shared_labels =  set(draw_set).intersection(set(gt_set))
        print(shared_labels)
        
        #Find the centers of each region in shared label
        draw_shared = self.find_label_centers(draw_smooth, shared_labels)
        gt_shared = self.find_label_centers(gt_smooth, shared_labels)
        
        #Find the centers of each region in unshared label
        draw_unshared = self.find_label_centers(draw_smooth, set(draw_set)-shared_labels)
        gt_unshared = self.find_label_centers(gt_smooth, set(gt_set)-shared_labels)

            
        #Resolve the unmatched pairs between draw_shared and gt_shared
        gt_draw_shared = self.pair_objects(draw_shared, gt_shared)
        
        #decouple the gt_draw_shared to a list of turples
        shared_item_list = []
        for key, value in gt_draw_shared.items():
            for d_center,gt_center in zip(value['draw_center'],value['gt_center']):
                shared_item_list.append((d_center, gt_center))
        
        #decouple the unshared objects to a list of turples
        unshared_item_list = []
        for key, value in draw_unshared.items():
            unshared_item_list += value
        for key, value in gt_unshared.items():
            unshared_item_list += value
            
        #compute the numerator score
        score = 0
        for x in range(len(shared_item_list)):
            for y in range(x+1, len(shared_item_list)):
                score += self.relevant_score(shared_item_list[x], shared_item_list[y])

        #compute the denomenator
        union = len(unshared_item_list)
        #union = len(draw_unshared) + len(gt_unshared)
        for key, value in gt_draw_shared.items():
            union += value['max_num_objects']
        intersection = len(shared_item_list)
        
        #print(score)
        if intersection > 2:
            final_score = score/(union*(intersection-1))
        elif intersection == 1:
            final_score = score/union
        else:
            final_score = 0
        return final_score
            
    def gaugancodraw_eval_metrics(self,label_d, label_gt, n_class, g_smooth=True, g_smooth_thred=1000):
        draw_smooth = label_d
        draw_smooth[np.where(draw_smooth == 147)] = 177
        draw_smooth[np.where(draw_smooth == 154)] = 177
        if g_smooth:
            g_unique_class, g_unique_counts = np.unique(label_gt, return_counts=True)
            #Need to smooth gt_labels as well

            result = np.where(g_unique_counts > g_smooth_thred) #3000 is a bit too much
            dominant_class = g_unique_class[result].tolist()
            gt_smooth = self.merge_noisy_pixels(label_gt, dominant_class)
            gt_smooth[np.where(gt_smooth == 147)] = 177
            gt_smooth[np.where(gt_smooth == 154)] = 177
        else:
            gt_smooth = label_gt
        #compute mean_IOU
        score_1 = self.mean_IoU(gt_smooth,draw_smooth, n_class)
        
        #compute relevant_score
        score_2 = self.relevant_eval_metrics(draw_smooth, gt_smooth)

        final_score = 2*score_1+3*score_2
        return final_score    

    def pair_objects(self, draw_shared, gt_shared):
        """
        Pair the regions in drawer's image and the regions in groundtruth image based on the mean square distance.
        TODO:
        1. Rethink the way we pair the objects
        """
        gt_draw_shared = {}

        for key in draw_shared.keys():
            #check if the number maps
            if len(draw_shared[key]) == len(gt_shared[key]) and len(draw_shared[key]) == 1:
                gt_draw_shared[key] = {"draw_center": draw_shared[key], "gt_center": gt_shared[key], "max_num_objects": 1}
            else:
                #pair the centers
                pair_regions = []
                pair_index = []
                for i, draw_c in enumerate(draw_shared[key]):
                    for j, gt_c in enumerate(gt_shared[key]):
                        pair_regions.append((draw_c, gt_c))
                        pair_index.append((i,j))
                #group all possible combinations of i,j together
                if len(draw_shared[key]) < len(gt_shared[key]):
                    perm = permutations(range(len(gt_shared[key])),len(draw_shared[key]))
                    pair_candidates = []
                    for p in perm:
                        #form the groups
                        pair_centers = []
                        for i in range(len(draw_shared[key])):
                            current_pair = (i,p[i])
                            current_pair_index = pair_index.index(current_pair)
                            current_pair_centers = pair_regions[current_pair_index]
                            pair_centers.append(current_pair_centers)
                        pair_candidates.append(pair_centers)
                else:
                    perm = permutations(range(len(draw_shared[key])),len(gt_shared[key]))
                    pair_candidates = []
                    for p in perm:
                        #form the groups
                        pair_centers = []
                        for i in range(len(gt_shared[key])):
                            current_pair = (p[i],i)
                            current_pair_index = pair_index.index(current_pair)
                            current_pair_centers = pair_regions[current_pair_index]
                            pair_centers.append(current_pair_centers)
                        pair_candidates.append(pair_centers)
                
                #sort pair_candidates based on their pair sum
                pair_candidates.sort(key = lambda p: sum([sqrt((c[0][0]-c[1][0])**2 + (c[0][1]-c[1][1])**2) for c in p]))
                
                optimal_pair = pair_candidates[0]
                paired_draw_centers = [x[0] for x in optimal_pair]
                paired_gt_centers = [x[1] for x in optimal_pair]

                gt_draw_shared[key] = {"draw_center": paired_draw_centers, "gt_center": paired_gt_centers, "max_num_objects": max(len(draw_shared[key]), len(gt_shared[key]))}
        return gt_draw_shared

if __name__ == '__main__':    
    ground_truth = Image.open('./test1.png').convert('L')
    if ground_truth.size[0] < ground_truth.size[1]:
        ground_truth = ground_truth.crop((0, 0, ground_truth.size[0], ground_truth.size[0])).resize((348, 350))
    else:
        ground_truth = ground_truth.crop((0, 0, ground_truth.size[1], ground_truth.size[1])).resize((348, 350))  
    (synthetic_img, scores) = Seg2Real().seg2real(ground_truth, Image.open('./test2.jpg'))
    print(scores)