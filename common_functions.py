from PIL import Image
import numpy as np
import diplib
from PIL import ImageFilter
import astroalign as aa
import cv2
import sys

def c(img):
    return Image.fromarray(img)

def adjust_contrast(img, min=2, max = 98):
    # pixvals = np.array(img)

    minval = np.percentile(img, min) # room for experimentation 
    maxval = np.percentile(img, max) # room for experimentation 
    img = np.clip(img, minval, maxval)
    img = ((img - minval) / (maxval - minval)) * 255
    return (img.astype(np.uint8))

def blur_noise_reduction(im):
    im = np.array(diplib.AreaOpening(im, filterSize=5, connectivity=2)) ## double chec kthis
    im1 = Image.fromarray(im)

    im1 = im1.filter(ImageFilter.BLUR)

    im1 = np.array(im1)
    # print(im1)
    im1 = adjust_contrast(im1, 95, 100)

    return im1

def threshold(im, percentile):
    p = np.percentile(im, percentile)
    im[im < p]  = 0
    im[im >= p]  = 255
    
    return im

def c_many(imgs, names):
    from matplotlib import pyplot as plt 

    rows = len(imgs)
    
    fig = plt.figure(figsize=(8*rows, 15*rows)) 

    for i, img in enumerate(imgs):
        fig.add_subplot(rows, rows, i+1) 

        plt.imshow(img) 
        plt.axis('off') 
        plt.title(names[i]) 
        
def tint(img, color):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
        i = np.array(img)
        
    if color == "blue":
        i[:,:,2] = 0
        i[:,:,1] = 0
    if color == "red":
        i[:,:,1] = 0
        i[:,:,0] = 0
    if color == "green":
        i[:,:,2] = 0
        i[:,:,0] = 0
        
    return Image.fromarray(i)

def overlay_images(img1, img2, apply_tint=False):
    conv = lambda img: (img).astype('uint8')
    im1 = Image.fromarray(conv(np.array(img1)))
    im2 = Image.fromarray(conv(np.array(img2)))
    
    if apply_tint:
        im1 = tint(im1, "blue")
        im2 = tint(im2, "red")
    
    return Image.blend(im1, im2, 0.5)

def qgg():
    pass

# the next few sections of code find center points

def get_center_points(mat_orig, min_size=0, filter=False, gray=True):
    mat = (mat_orig == 255)
    gray_edit_copy = mat_orig.copy() if gray else None
    filtered = mat_orig.copy() if filter else None

    def fill_blob(x, y):
        
        mat[y][x] = False
        blob_elems = [[x, y]]
        #  right
        if  (x != mat.shape[1] - 1) and mat[y][x+1]:
            blob_elems +=(fill_blob(x+1, y))
        # down
        if  (y != mat.shape[0] - 1) and mat[y+1][x]:
            blob_elems +=(fill_blob(x, y+1))
        # left
        if  (x != 0) and mat[y][x-1]:
            blob_elems +=(fill_blob(x-1, y))
        # up
        if  (y != 0) and  mat[y-1][x]:
            blob_elems +=(fill_blob(x, y-1))
        
        return blob_elems

    
    midpoints = []
    sizes = []
    for y, line in enumerate(mat):
        for x, pixel in enumerate(line):
            if pixel:
                points = fill_blob(x, y)
                if len(points) > min_size :       
                    midpoints.append([ round(x) for x in list(np.average(points, axis=0))])
                    sizes.append(len(points))
                else:
                    if filter:
                        for p in points:
                            filtered[p[1]][p[0]] = 0


    if gray:
        gray_edit_copy = gray_edit_copy.__invert__()
        for i in midpoints:
            gray_edit_copy[i[1], i [0]] = 170
            
            # gray_edit_copy[i[1] - 1, i [0] -1 ] = 170
            # gray_edit_copy[i[1] + 1, i [0] + 1] = 170
            # gray_edit_copy[i[1] + 1, i [0] -1 ] = 170
            # gray_edit_copy[i[1] - 1, i [0] + 1] = 170

    if filter:
        return midpoints, sizes, gray_edit_copy, filtered
    return midpoints, sizes, gray_edit_copy
    
        



def get_blobs(mat_orig):
    mat = (mat_orig == 255)
    gray_edit_copy = mat_orig.copy()

    def fill_blob(x, y):
        
        mat[y][x] = False
        blob_elems = [[x, y]]
        #  right
        if  (x != mat.shape[1] - 1) and mat[y][x+1]:
            blob_elems +=(fill_blob(x+1, y))
        # down
        if  (y != mat.shape[0] - 1) and mat[y+1][x]:
            blob_elems +=(fill_blob(x, y+1))
        # left
        if  (x != 0) and mat[y][x-1]:
            blob_elems +=(fill_blob(x-1, y))
        # up
        if  (y != 0) and  mat[y-1][x]:
            blob_elems +=(fill_blob(x, y-1))
         
        return blob_elems

    
    blobs = []
    for y, line in enumerate(mat):
        for x, pixel in enumerate(line):
            if pixel:
                points = fill_blob(x, y)
                blobs.append(points)

    return blobs





def draw_gray_x(gray_edit_copy, midpoints):
    gray_edit_copy = gray_edit_copy.copy()
    for i in midpoints:
        gray_edit_copy[i[1], i [0]] = 170
        gray_edit_copy[i[1] - 1, i [0] -1 ] = 170
        gray_edit_copy[i[1] + 1, i [0] + 1] = 170
        gray_edit_copy[i[1] + 1, i [0] -1 ] = 170
        gray_edit_copy[i[1] - 1, i [0] + 1] = 170
    return gray_edit_copy

def draw_gray_t(gray_edit_copy, midpoints ):
    gray_edit_copy = gray_edit_copy.copy()
    color = 200
    for i in midpoints:
        gray_edit_copy[i[1], i [0]] = color
        gray_edit_copy[i[1] , i [0] -1 ] = color
        gray_edit_copy[i[1], i [0] + 1] = color
        gray_edit_copy[i[1] + 1, i [0] ] = color
        gray_edit_copy[i[1] - 1, i [0]] = color
    return gray_edit_copy

def lower_right(midpoints):
    current_max = 0
    pt = []
    for i, point in enumerate(midpoints):
        if np.sum(point) > current_max:
            current_max = np.sum(point)
            pt = point
            
    # print(current_max, pt)
    
    return pt

def upper_left(midpoint):
    current_min = 99999999
    ptmin = []
    for i, point2 in enumerate(midpoint):
        if np.sum(point2) <= current_min:
            current_min = np.sum(point2)
            ptmin = point2
            
    # print(current_min, ptmin)
    
    return ptmin