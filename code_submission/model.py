"""
Exemplary predictive model.

You must provide at least 2 methods:
- __init__: Initialization of the class instance
- predict: Uses the model to perform predictions.

The following convenience methods are provided:
- load_microwave_volume: Load three-dimensional microwave image
- visualize_microwave_volume: Visualize the slices of a three-dimensional microwave image
"""

import os
import numpy as np
import json
import glob
import skimage.io
from matplotlib import pyplot as plt
import skimage.io
import cv2
from sklearn.cluster import KMeans
from skimage import feature
from skimage.feature import corner_harris, corner_subpix, corner_peaks

class Model:
    """
    Rohde & Schwarz Engineering Competition 2022 class template
    """
    def __init__(self):
        """
        Initialize the class instance

        Important: If you want to refer to relative paths, e.g., './subdir', use
        os.path.join(os.path.dirname(__file__), 'subdir')
        
        """
    def predict(self, data_set_directory):
        """
        This function should provide predictions of labels on a data set.
        Make sure that the predictions are in the correct format for the scoring metric. The method should return an
        array of dictionaries, where the number of dictionaries must match the number of tiff files in
        data_set_dictionary.
        """
        predictions = []
        input_files = glob.glob(os.path.join(os.path.abspath(data_set_directory), '*.tiff'))
        for i in range(len(input_files)):
            img = self.read_img1(input_files[i])
            img2 = self.preprocess1(img)
            rect1, M1 = self.detect_rect(img2)
            points1, num_pills1 = self.detect_points(rect1)
            missing_pills1, present_pills1 = self.pill_present_detection(points1, rect1, num_pills1)
            pill_dict1 = self.transformed_points(missing_pills1, present_pills1, M1)
            label = {}
            label["file"] = os.path.basename(input_files[i])
            label["coordinates"] = pill_dict1
            label["missing_pills"] = len(missing_pills1)
            label["present_pills"] = len(present_pills1)
            predictions.append(label)

        # return list of dictionaries whose length matches the number of tiff files
        return predictions
    
    
    def read_img1(self, tiff_path):

        img = skimage.io.imread(tiff_path)
        rgbArray1 = np.full((257,257,3),0, 'uint8')
        for i in range(img.shape[2]):
            rgbArray1[..., i] = 3 * np.log10(img[:, :, i])
        return rgbArray1

    def preprocess1(self, rgbArray1):
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(rgbArray1.reshape(-1, 3))
        res = kmeans.labels_.reshape(257, 257)
        res = 255 * res
        counts, bin = np.histogram(res.flatten())
        if counts[0] > counts[-1]:
            a = 255 * np.ones((257,257))
            res = a - res
        res = res.astype('u1')       
        return res

    def detect_rect(self, res):
        edges1 = feature.canny(res.reshape(257,257), sigma = 10)
        coords = corner_peaks(corner_harris(edges1), min_distance=1, threshold_rel=0.1)
        coords[:,[0, 1]] = coords[:,[1, 0]]
        
        rect = cv2.minAreaRect(coords)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        top_left_x = min(box[:,1])
        top_left_y = min(box[:,0])
        bot_right_x = max(box[:,1])
        bot_right_y = max(box[:,0])

        rect = cv2.minAreaRect(coords)
        box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(res, M, (width, height))

        # cv2.imwrite("crop_img.jpg", warped)
        
        return warped, M


    # In[226]:


    def detect_points(self, warped):

        shape = warped.shape

        if shape[0] > shape[1]:
            vertical = 1 #detect orientation
        else:
            vertical = 0

        max_shape = max(shape)

        if max_shape < 141:
            num_pills = 10
            long_dim = 135
            short_dim = 90
            x_points = [25, 70]
            y_points = [20, 45, 67, 87, 110]

        else:
            num_pills = 15
            long_dim = 150
            short_dim = 90
            y_points = [20, 45, 70, 95, 120]
            x_points = [20, 45, 70]

        if vertical == 1:
            dim = (short_dim, long_dim)
        else:
            dim = (long_dim, short_dim)
            x_points, y_points = y_points, x_points

        resized = cv2.resize(warped, dim, interpolation = cv2.INTER_AREA)    
        #plt.imshow(resized)
        points = np.array(np.meshgrid(x_points, y_points)).T.reshape(-1,2)
        #plt.scatter(points[:,0], points[:,1], color = 'r')
        
        return points, num_pills


    # In[246]:


    def pill_present_detection(self, points, rect, num_pills):

        present_pills = []
        missing_pills = []

        for i in range(0, num_pills):
            x, y = points[i]

            x1 = x-9
            x2 = x+9

            y1 = y-9 
            y2 = y+9

            hist, bin_edges = np.histogram(rect[y1:y2, x1:x2].flatten())

            if hist[0] > hist[-1] + 25:
                present_pills.append(points[i])
            else: 
                missing_pills.append(points[i])

        missing_pills = np.array(missing_pills)
        present_pills = np.array(present_pills)  
        
        return missing_pills, present_pills

    def transformed_points(self, missing_pills, present_pills, M):
        
        dict_coords= {}
        
        if len(missing_pills) != 0:
            s1 = tuple(missing_pills[:,1])
            s2 = tuple(missing_pills[:,0])
            s  =(s1,s2)
            new = Model.transform_coordinates_back(s, M)
            tra_missing = np.vstack((new[0], new[1])).T
            tra_missing[:, [0, 1]] =  tra_missing[:,[1, 0]]
            
            dummy1 = 257 * np.ones(tra_missing.shape[0])
            tra_missing[:, 1] = dummy1 - tra_missing[:, 1]
            
            dict_coords["missing"] = tra_missing
        else:
            dict_coords["missing"] = []
                    

        if len(present_pills) != 0:
            s1 = tuple(present_pills[:, 1])
            s2 = tuple(present_pills[:, 0])
            s  = (s1, s2)
            new = Model.transform_coordinates_back(s, M)
            tra_present = np.vstack((new[0], new[1])).T
            tra_present[:, [0, 1]] =  tra_present[:, [1, 0]]
            
            dummy2 = 257*np.ones(tra_present.shape[0])
            tra_present[:,1] = dummy2 - tra_present[:,1]
            
            dict_coords["present"] = tra_present
        else:
            dict_coords["present"] = []
        

        return dict_coords

    def transform_coordinates_back(idcs, M):
        """Undoes the perspective transformation given by M on a tuple of indices idcs. idcs must have the following shape ((x1, x2, x3, ...), (y1, y2, y3, ...))"""
        # Flip and concatenate with a number 1 (it is assumed that t_i = 1 in equation in https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga20f62aa3235d869c9956436c870893ae)
        # may be a source of the error
        idcs1 = np.concatenate((idcs[ : : -1], np.ones((1, len(idcs[0])))), axis=0)

        # Multiply with the inverse of the perspective transformation matrix
        idcs_dst = np.round(np.linalg.inv(M)@idcs1)

        # Convert to a list of tuples and flip
        idcs_dst = [tuple(lst) for lst in idcs_dst[:2].astype('u1').tolist()][::-1]
        return tuple(idcs_dst)
