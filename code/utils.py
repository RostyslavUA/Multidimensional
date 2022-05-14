import os.path as osp
import json
import numpy as np
import torch
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

# Deprecated. See new mask_pic function below
# def mask_pic(coord, pic_size=(257, 257)):  # Hard-coded size of the picture  
#     """Function generates a picture mask from the missing and present pills' coordinates. When casting a picture to a matrix, the first-row, first-column entry represents a top-left pixel of the picture. It corresponds to the coordinate (0, 0). But the (0, 0) coordinate of missing/present pills' given in .tiff files represents the bottom-left pixel. This means that we are dealing with two different coordinate systems. The transformations present in this function ensure that the coordinates of the pic and the resulting mask have one-to-one correspondence"""
    
#     timg = torch.full(pic_size, 0).type(torch.float)  # Ternary image

#     pixels = torch.round(coord).type(torch.long)  # Rounded coordinate values of missing/present pills, correspond to the pixels they are shown at

#     pixels_m = pixels[:, :2].unique(dim=0)  # Missing pills' coordinates
#     pixels_m[:, 1] = 257 - pixels_m[:, 1]  # Y-coord is flipped

#     pixels_p = pixels[:, 2:].unique(dim=0)  # Present pills' coordinates
#     pixels_p[:, 1] = 257 - pixels_p[:, 1] 

#     # Set the values of the pixels corresponding to missing pills to 0.5
#     # I believe, there is a more elegant way of doing so
#     for pixels in pixels_m:
#         if pixels.tolist() != [0, 257]:  # 0, 257 isn't a coordinate. It means that pill is neither missing nor present. Maybe a pitfall if the missing/present pills is shown exactly at the coordinate (0, 0) (coordinate system in .tiff)
#             timg[tuple(pixels)] = 0.5

#     # Set the values of the pixels corresponding to present pills to 1
#     for pixels in pixels_p:
#         if pixels.tolist() != [0, 257]:  
#             timg[tuple(pixels)] = 1

#     return timg.T.unsqueeze(0)


def transform_coordinates_back(idcs, M):
    """Undoes the perspective transformation given by M on a tuple of indices idcs. idcs must have the following shape ((x1, x2, x3, ...), (y1, y2, y3, ...))"""
    # Flip and concatenate with a number 1 (it is assumed that t_i = 1 in equation in https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga20f62aa3235d869c9956436c870893ae)
    # may be a source of the error
    idcs1 = np.concatenate((idcs[::-1], np.ones((1, len(idcs[0])))), axis=0)

    # Multiply with the inverse of the perspective transformation matrix
    idcs_dst = np.round(np.linalg.inv(M)@idcs1)

    # Convert to a list of tuples and flip
    idcs_dst = [tuple(lst) for lst in idcs_dst[:2].astype('u1').tolist()][::-1]
    return tuple(idcs_dst)
    
    
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
    _coor = np.array(coor)
    _coor[:, 1] = 257 - _coor[:, 1]  # Flipped y-axis
    coor_adj = tuple(tuple(val) for val in _coor)  # Coordinae adjusted
    return coor_adj


def transform_coordinates(coord, M, was_vertical):
    """Transform coordinates to match the perspective of the cropped image. Rostyslav keeps all crops in a horizontal position
    i.e., height<width. Thus, if you had to transform the crop from vertical to horizontal position, set was_vertical=True. 
    The output of this function you will probably want to propagate through mask_pic function to obtain a mask"""
    coord_cat = np.concatenate((coord, np.ones((len(coord), 1))), -1)  # Coordinates present concatenated
    _coord_t = np.round(abs(M@np.expand_dims(coord_cat, 2))).squeeze(-1)[:, :2].astype('int32')  # Coordinates present transformed
    # Cast to tuples
    coord_t = tuple(zip(*[tuple(val) for val in _coord_t]))
    return coord_t if was_vertical else coord_t[::-1]


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