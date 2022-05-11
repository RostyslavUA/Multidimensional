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


def mask_pic(coord, pic_size=(257, 257)):  # Hard-coded size of the picture
    """Function generates a picture mask from the missing and present pills' coordinates. When casting a picture to a matrix, the first-row, first-column entry represents a top-left pixel of the picture. It corresponds to the coordinate (0, 0). But the (0, 0) coordinate of missing/present pills' given in .tiff files represents the bottom-left pixel. This means that we are dealing with two different coordinate systems. The transformations present in this function ensure that the coordinates of the pic and the resulting mask have one-to-one correspondence"""
    
    timg = torch.full(pic_size, 0).type(torch.float)  # Ternary image

    pixels = torch.round(coord).type(torch.long)  # Rounded coordinate values of missing/present pills, correspond to the pixels they are shown at

    pixels_m = pixels[:, :2].unique(dim=0)  # Missing pills' coordinates
    pixels_m[:, 1] = 257 - pixels_m[:, 1]  # Y-coord is flipped

    pixels_p = pixels[:, 2:].unique(dim=0)  # Present pills' coordinates
    pixels_p[:, 1] = 257 - pixels_p[:, 1] 

    # Set the values of the pixels corresponding to missing pills to 0.5
    # I believe, there is a more elegant way of doing so
    for pixels in pixels_m:
        if pixels.tolist() != [0, 257]:  # 0, 257 isn't a coordinate. It means that pill is neither missing nor present. Maybe a pitfall if the missing/present pills is shown exactly at the coordinate (0, 0) (coordinate system in .tiff)
            timg[tuple(pixels)] = 0.5

    # Set the values of the pixels corresponding to present pills to 1
    for pixels in pixels_p:
        if pixels.tolist() != [0, 257]:  
            timg[tuple(pixels)] = 1

    return timg.T.unsqueeze(0)


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
    return pxs_missing, pxs_present, pxs_missing_sum, pxs_present_sum