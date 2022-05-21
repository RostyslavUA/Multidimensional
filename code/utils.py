import os.path as osp
import json
import numpy as np
import torch
import cv2
from skimage.feature import canny, corner_harris, corner_peaks
from scipy import fft
from PIL import Image
import matplotlib.pyplot as plt


def process_coord(_coord_missing, _coord_present):
    """Functions reforms xy coodrinates from a list of lists into a ndarray. The coordinates are zero-padded if necessary
    to obtain all dimensions of the same shape. Last dim of returned array contains 2 pairs of xy coordinates: 
    1st pair - missing pills, 2nd pair - present pills"""
    max_length = max(max([len(coord) for coord in _coord_present]), max([len(coord) for coord in _coord_missing]))
    coord_missing = np.array([np.pad(coord, [(0, max_length-len(coord)), (0, 0)], constant_values=0) if len(coord) 
                              else np.zeros((max_length, 2)) 
                              for coord in _coord_missing])
    coord_present = np.array([np.pad(coord, [(0, max_length-len(coord)), (0, 0)], constant_values=0) if len(coord) 
                              else np.zeros((max_length, 2)) 
                              for coord in _coord_present])
    coord = np.concatenate((coord_missing, coord_present), axis = -1)
    return coord


def process_labels(path, label_names):
    """Function extracts the coordinates and the number of missing pills from the labels"""
    _coord_missing, _coord_present, _pills_missing, _pills_present = [], [], [], []
    for name in label_names:
        with open(osp.join(path, name)) as f:
            d = json.load(f)
            _coord_missing.append(d['coordinates']['missing'])
            _coord_present.append(d['coordinates']['present'])
            _pills_missing.append(d['missing_pills'])
            _pills_present.append(d['present_pills'])
    return _coord_missing, _coord_present, _pills_missing, _pills_present


def process_samples(path, sample_names):
    """Function reads the images and returns a 3d array of shape (number of images, shape along x axis, shape along y axis)"""
    
    _samples = []
    for name in sample_names:
        _samples.append(np.array(Image.open(osp.join(path, name))))
    return _samples

def transform_coordinates(coord, M, was_vertical):
    """Transform coordinates to match the perspective of the cropped image. Rostyslav keeps all crops in a horizontal position
    i.e., height<width. Thus, if you had to transform the crop from vertical to horizontal position, set was_vertical=True. 
    The output of this function you will probably want to propagate through mask_pic function to obtain a mask"""
    if len(coord) == 0:
        # No missing/present coordinates
        return tuple()
    coord_cat = np.concatenate((coord, np.ones((len(coord), 1))), -1)  # Coordinates present concatenated
    _coord_t = np.round(abs(M@np.expand_dims(coord_cat, 2))).squeeze(-1)[:, :2].astype('int32')  # Coordinates present transformed
    # Cast to tuples
    coord_t = tuple(zip(*[tuple(val) for val in _coord_t]))
    return coord_t if was_vertical else coord_t[::-1]


def transform_coordinates_back(idcs, M, was_vertical):
    """Undoes the perspective transformation given by M on a tuple of indices idcs. idcs must have the following shape ((x1, x2, x3, ...), (y1, y2, y3, ...))"""
        # Flip and concatenate with a number 1 (it is assumed that t_i = 1 in equation in https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga20f62aa3235d869c9956436c870893ae)
    # may be a source of the error
    if len(idcs)==0:
        # idcs is empty
        return ()
    idcs1 = np.concatenate((idcs[::-1], np.ones((1, len(idcs[0])))), axis=0) if not was_vertical else np.concatenate((idcs, np.ones((1, len(idcs[0])))), axis=0)
    
    # Multiply with the inverse of the perspective transformation matrix
    idcs_dst = np.round(np.linalg.inv(M)@idcs1)
    # Convert to a list of tuples and flip
    idcs_dst = [tuple(lst) for lst in idcs_dst[:2].astype('int32').tolist()]
    idcs_dst = tuple(idcs_dst) if was_vertical else tuple(idcs_dst)[::-1]  # return was here
    if was_vertical and len(idcs_dst) > 0:
        idcs_dst = idcs_dst[::-1]
    # Adjust the coordinates so that they match provided labels
    predicted_coordinates = np.array(list(zip(*idcs_dst)))
    if predicted_coordinates.size:
        predicted_coordinates[:, 0] = 257 - predicted_coordinates[:, 0]
        predicted_coordinates[:, [0, 1]] = predicted_coordinates[:, [1, 0]]
        predicted_coordinates = predicted_coordinates.tolist()
    else:
        predicted_coordinates = predicted_coordinates.tolist()
    return predicted_coordinates
    
def get_px_values(img, coord_miss, coord_pres, width=10, height=10, plot=False):
    """Function returns the pixel values around missing and present pills and the sum of some set of pixels around missing and
    present pills. Plotting option is available"""
    # Normalize to the sum intensity of the picture
    img[:, :, 0] = img[:, :, 0]/img[:, :, 0].sum()
    _coord_missing, _coord_present = [], []
    _pxs_missing, _pxs_present = [], []
    _pxs_missing_sum, _pxs_present_sum = [], []
    for coord in coord_miss:
        _coord_missing.append((slice(257-coord[1]-height//2, 257-coord[1]+height//2), slice(coord[0]-width//2, coord[0]+width//2), 0))
        _pxs_missing.append(img[_coord_missing[-1]].flatten())  # We do not need to keep dimensionality anymore  
        _pxs_missing_sum.append(img[_coord_missing[-1]].flatten().sum())
    for coord in coord_pres:
        _coord_present.append((slice(257-coord[1]-height//2, 257-coord[1]+height//2), slice(coord[0]-width//2, coord[0]+width//2), 0))
        _pxs_present.append(img[_coord_present[-1]].flatten())
        _pxs_present_sum.append(img[_coord_present[-1]].flatten().sum())
        
    if plot:
        _img = img.copy()
        for coord_slice in _coord_missing:
            _img[coord_slice] = 2  # Set to some value to see the difference
        for coord_slice in _coord_present:
            _img[coord_slice] = 1  # Set to smaller value to see the dfiiference between missing and present pills
            plt.imshow(_img[:, :, 0], cmap='bwr')
    pxs_missing = np.concatenate(_pxs_missing).flatten() if len(_pxs_missing) else np.array([])
    pxs_present = np.concatenate(_pxs_present).flatten() if len(_pxs_present) else np.array([])
    
    pxs_missing_sum = np.array(_pxs_missing_sum).flatten() if len(_pxs_missing_sum) else np.array([])
    pxs_present_sum =  np.array(_pxs_present_sum).flatten() if len(_pxs_present_sum) else np.array([])
    return pxs_missing, pxs_present, pxs_missing_sum/(width*height), pxs_present_sum/(width*height)


def swindow(arr, width=2, height=2, padding=True, normalize=True):
    """Sliding window over a ndarray. For every new position of the window, saves a sum of the px values into _sum.
    To preserve dimensionality, zero-padding option is available. Normalization option is available too"""
    arr = arr/arr.sum() if normalize else arr
    if padding:
        arr = np.concatenate((arr, np.zeros((arr.shape[0], 1))), axis=-1)  # Concatenate a column from a right
        arr = np.concatenate((arr, np.zeros((1, arr.shape[-1]))), axis=0)  # Concatenate a colum below
    _sum = []
    for i in range(arr.shape[0] - 1):
        for j in range(arr.shape[-1] - 1):
            _sum.append(arr[i:i+width, j:j+height].mean())  # temporarily replace by mean
    summask = np.array(_sum).reshape(arr.shape[0]-1, arr.shape[-1]-1)
     # Normalize to the pic size if needed
    return summask/(width*height) if normalize else summask


def swindow_mask(img, w=2, h=2):
    """Apply a sliding window on the three slices, compute its mean and classify the pixels based on their values"""
    mask1, mask2, mask3 = swindow(img[:, :, 0], w, h), swindow(img[:, :, 1], w, h), swindow(img[:, :, 2], w, h)
    masks_mean = np.array([mask1, mask2, mask3]).mean(axis=0)
    # Numbers are coming from the observed distributions of the pixel values of the missing/present pills
    masks_mean[np.logical_or(masks_mean<0.4e-5, masks_mean>9e-5)] = -1  # Blister and empty space
    masks_mean[np.logical_and(masks_mean>0.4e-5, masks_mean<2.5e-5)] = 0  # Missing pill
    masks_mean[np.logical_and(masks_mean>2.5e-5, masks_mean<9e-5)] = 1  # Present pill
    return masks_mean


def adjust_coordinates(coor):
    """Flip coordinates (labels provided by R&S) and cast them to tuple. You can use it direcly after uploading the labels 
    i.e., adjust_coordinates(d['coordinates']['present']). Output of this function is then supplied into transform_coordinates
    function"""
    if not len(coor):
        # No missing/present coordinates
        return tuple()
    _coor = np.array(coor)
    _coor[:, 1] = 257 - _coor[:, 1]  # Flipped y-axis
    coor_adj = tuple(tuple(val) for val in _coor)  # Coordinae adjusted
    return coor_adj


def mask_pic(img, missing_coordinates, present_coordinates, height=10, width=10, miss_val=0, pres_val=1, backg_val=0.5):
    _img = img.copy()  # Make a deep copy
    for val in zip(*missing_coordinates):
        c1, c2 = val
        _img[c1-height//2:c1+height//2, c2-width//2:c2+width//2] = miss_val
    for val in zip(*present_coordinates):
        c1, c2 = val
        _img[c1-height//2:c1+height//2, c2-width//2:c2+width//2] = pres_val
    _img[np.logical_and(_img!=miss_val, _img!=pres_val)] = 0.5  # Background and blister
    return _img

def crop(img):
    """Crop an image. Returns the cropped image, used transformarion matrix and whether the picture had to be rotated to make it horizontal """
    # Detect edges
    img_m = cv2.medianBlur(img, 5)
    edges1 = canny(img_m[:, :, 0], sigma = 1, low_threshold=0.7, high_threshold=1)

    # Get the coordinates of the corner peaks
    coords = corner_peaks(corner_harris(edges1), min_distance=1, threshold_rel=0.1)

    # Flip these coordinates
    coords[:,[0, 1]] = coords[:,[1, 0]]

    rect = cv2.minAreaRect(coords)  # Fit the rectangle in the peak coordinates
    box = cv2.boxPoints(rect)  # Get 4 (x, y) coordinates of the rectangle
    box = np.int0(box)

    res1d = img[:, :, 0].copy()  # Make a deep copy

    cv2.drawContours(res1d, [box], 0, (255,255,255), 2)  # Draw contours 


    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # Perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    was_vertical=False
    if warped.shape[0] > warped.shape[1]:
        was_vertical = True  # Image had vertical orientation. Will have to be considered by coordinate transformation
        warped = warped.transpose(1, 0, 2)  # Ensure horizontal orientation
    return warped, M, was_vertical

# Deprecated, since it has too many local variables. Replaced by crop_pills_and_save_coordinates & crop_pills_and_save_stats
# def crop_pill(img, img_copy, save_for_histo=False, plot=False):
#     """Functions extracts the pills from the cropped blister. If img_copy is used to collect the values for histogram,
#      it should already contain the pixel's with the known values (usually 10 for present pill). save_for_histo option
#      saves the cropped pills into the lists of missing/present pills. Plotting option is available."""
    
#     img = normalize_pic(img)
#     img = non_lin_transform(img)

#     idx_col_center = img.shape[1]//2
#     width = img.shape[1]//10
#     shift_col = 2*(img.shape[1]//10-1)

#     nr_rows = 2 if img.shape[1] < 135 else 3  # 2 or 3 rows in the blister
#     idx_row_center = img.shape[0]//nr_rows

#     # Lists for histograms
#     _miss_px_vals = []
#     _pres_px_vals = []

#     # Lists for saving coordinates
#     _coorm = []
#     _coorp = []
#     for i in range(nr_rows):
#         for j in range(-2, 3):
#             # Extract the non-linearly-transformed pills
#             pill_cell = img[i*idx_row_center:(i+1)*idx_row_center,
#                                idx_col_center-width+j*shift_col:idx_col_center+width+j*shift_col]
#             if pill_cell.flatten().mean() < -0.6:
#                 _coorm.append((pill_cell.shape[0]//2+i*idx_row_center, idx_col_center+j*shift_col))  # Predicted missing
#             else:
#                 _coorp.append((pill_cell.shape[0]//2+i*idx_row_center, idx_col_center+j*shift_col))  # Predicted present
#             if save_for_histo:
#             # Save the crops of the pills for plotting histogram
#                 if (img_copy[i*idx_row_center:(i+1)*idx_row_center,
#                     idx_col_center-width+j*shift_col:idx_col_center+width+j*shift_col] == 10).any():
#                     # If any of pixel values equals 10 (we have set it intentionally previously), it is a present pill
#                     _pres_px_vals.append(pill_cell.flatten())
#                 else:
#                     _miss_px_vals.append(pill_cell.flatten())

#             if plot:
#                 # Plot a cropped pill and the corresponding histogram. Note that histogram values are after non-linear
#                 # transformation
#                 _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
#                 ax1.imshow(img_copy[i*idx_row_center:(i+1)*idx_row_center,
#                                    idx_col_center-width+j*shift_col:idx_col_center+width+j*shift_col])
#                 ax2.hist(pill_cell.flatten(), color='brown', label='Pixel intensities of a cropped pill')
#                 #ax2.set_xlim([-3, 4])
#                 #ax2.set_ylim([0, 1000])
#                 plt.legend()
#                 #plt.savefig('../status/crop_and_histo.jpg')
#                 plt.show()
#     return _coorm, _coorp, np.array(_miss_px_vals).flatten(), np.array(_pres_px_vals).flatten()

def normalize_pic(img):
    """Function normalizes a each slice to its Frobenius norm with subsequent transformation of mean and std. The resulting
    slice's pixel distribution has the following parameters: mean = 0 and std = 1"""
    img_norm = img/(np.linalg.norm(img, 'fro', axis=(0, 1)))  # Frobenius norm
    img_norm = img_norm - img_norm.mean(axis=(0, 1))
    img_norm = img_norm*1/img_norm.std(axis=(0, 1))
    return img_norm

def non_lin_transform(img):
    """Function applies non-linear transformation to make the missing and present pill's distributions linearly separable"""
    img_nl = img**3  # Apply a nonlinear transformation to separate the values of missing and present pills
    return img_nl

def pill_cropping_pars(img):
    """Function returns pill cropping parameters that depend on the image size"""
    idx_col_center = img.shape[1]//2
    width = img.shape[1]//10
    shift_col = 2*(img.shape[1]//10-1)
    nr_rows = 2 if img.shape[1] < 135 else 3  # 2 or 3 rows in the blister
    idx_row_center = img.shape[0]//nr_rows
    return idx_col_center, width, shift_col, nr_rows, idx_row_center

def crop_pills_and_save_coordinates(img):
    """Functions extracts the pills from the cropped blister. If img_copy is used to collect the values for histogram,
     it should already contain the pixel's with the known values (usually 10 for present pill). save_for_histo option
     saves the cropped pills into the lists of missing/present pills. Plotting option is available."""
    idx_col_center, width, shift_col, nr_rows, idx_row_center = pill_cropping_pars(img)
    # Lists for saving coordinates
    _coorm, _coorp = [], []
    for i in range(nr_rows):
        for j in range(-2, 3):
            # Crop the pills
            pill_cell = img[i*idx_row_center:(i+1)*idx_row_center,
                               idx_col_center-width+j*shift_col:idx_col_center+width+j*shift_col]
            if pill_cell.flatten().mean() < -0.6:
                # Predicted missing
                _coorm.append((pill_cell.shape[0]//2+i*idx_row_center, idx_col_center+j*shift_col))  
            else:
                # Predicted present
                _coorp.append((pill_cell.shape[0]//2+i*idx_row_center, idx_col_center+j*shift_col))  
    return _coorm, _coorp

def crop_pills_and_save_stats(img, coorp_t, plot=False):
    """Function collects the statistic of the pixel values associated with the missing/present pills. At the input the 
    function expect the cropped image and the correspondingly transformed coordinates of the present pills. The pixel's values
    associated with these coordinates will be set to 10. By doing so, we will be able to classify missing and present pills.
    Plotting option is available"""
    # Make a deep copy (changing a copy won't change the original img)
    img_copy = img.copy()
    # Set to a known pixel value, so that later we can identify whether the pill is missing or present
    img_copy[coorp_t] = 10
    idx_col_center, width, shift_col, nr_rows, idx_row_center = pill_cropping_pars(img)
    # Lists for histograms
    _miss_px_vals = []
    _pres_px_vals = []
    for i in range(nr_rows):
        for j in range(-2, 3):  # 5 columns
            # Crop of a pill's cell
            pill_cell = img[i*idx_row_center:(i+1)*idx_row_center,
                               idx_col_center-width+j*shift_col:idx_col_center+width+j*shift_col]
            if (img_copy[i*idx_row_center:(i+1)*idx_row_center,
                idx_col_center-width+j*shift_col:idx_col_center+width+j*shift_col] == 10).any():
                # If any of pixel values equals 10 (we have set it intentionally previously), it is a present pill
                _pres_px_vals.append(pill_cell.flatten())
            else:
                _miss_px_vals.append(pill_cell.flatten())
            if plot:
                # Plot a cropped pill and the corresponding histogram. Note that histogram values are after non-linear
                # transformation
                _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                ax1.imshow(img_copy[i*idx_row_center:(i+1)*idx_row_center,
                                   idx_col_center-width+j*shift_col:idx_col_center+width+j*shift_col])
                ax2.hist(pill_cell.flatten(), color='brown', label='Pixel intensities of a cropped pill')
                #ax2.set_xlim([-3, 4])
                #ax2.set_ylim([0, 1000])
                plt.legend()
                #plt.savefig('../status/crop_and_histo.jpg')
                plt.show()
    return np.array(_miss_px_vals).flatten(), np.array(_pres_px_vals).flatten()

def transform_label_coordinates(missing_coordinates, present_coordinates, M, was_vertical):
    """Function thransforms the coordinates of the provided labels to the coordinate frame given by the crop"""
    coorm = adjust_coordinates(missing_coordinates)
    coorp = adjust_coordinates(present_coordinates)
    coorm_t = transform_coordinates(coorm, M, was_vertical)  # _t stands for tansformed
    coorp_t = transform_coordinates(coorp, M, was_vertical)
    return coorm_t, coorp_t