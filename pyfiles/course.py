
from PIL import Image
import numpy as np
import diplib
from PIL import ImageFilter
import astroalign as aa
import cv2
import sys
sys.setrecursionlimit(53000) # override needed for computing midpoints, which uses a recursive function
Image.MAX_IMAGE_PIXELS = 366498276 # override is needed, or else it gives a DecompressionBombError


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

def get_center_points(mat_orig, min_size=0):
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

    
    midpoints = []
    sizes = []
    for y, line in enumerate(mat):
        for x, pixel in enumerate(line):
            if pixel:
                points = fill_blob(x, y)
                if len(points) > min_size :
                    midpoints.append([ round(x) for x in list(np.average(points, axis=0))])
                    sizes.append(len(points))



    gray_edit_copy = gray_edit_copy.__invert__()
    for i in midpoints:
        gray_edit_copy[i[1], i [0]] = 170
        gray_edit_copy[i[1] - 1, i [0] -1 ] = 170
        gray_edit_copy[i[1] + 1, i [0] + 1] = 170
        gray_edit_copy[i[1] + 1, i [0] -1 ] = 170
        gray_edit_copy[i[1] - 1, i [0] + 1] = 170

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
    # im1 = im1.filter(ImageFilter.BLUR)
    # im1 = im1.filter(ImageFilter.BLUR)
    # im1 = im1.filter(ImageFilter.BLUR)
    # im1 = im1.filter(ImageFilter.BLUR)
    # im1 = im1.filter(ImageFilter.BLUR)
    # im1 = im1.filter(ImageFilter.BLUR).filter(ImageFilter.BLUR).filter(ImageFilter.BLUR).filter(ImageFilter.BLUR).filter(ImageFilter.BLUR).filter(ImageFilter.BLUR)
    # im1 = im1 - np.array(diplib.AreaOpening(np.array(im1), filterSize=25, connectivity=2))
    
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




## open tiffs
## lower res

layer_to_align = 2

img_cy1 = Image.open('originals/ImageSLIDE4-CYCLE1.tif')
img_cy1.seek(layer_to_align) # navigate to brightfield
brightfield_cy1 = np.array(img_cy1)
brightfield_cy1 = (brightfield_cy1/256).astype('uint8') # don't need the whole range of values, so this reduces mem size, + improves access speed

img_cy2 = Image.open('originals/ImageARRAY4-CYCLE2.tif')
img_cy2.seek(layer_to_align) # navigate to brightfield
brightfield_cy2 = np.array(img_cy2)
brightfield_cy2 = (brightfield_cy2/256).astype('uint8') 



# The code seeks to fix the following problem:
# PROBLEM: the two images have different sizes, so when you attempt to scale them down, they get squished
y_cf1, x_cf1 = brightfield_cy1.shape
y_cf2, x_cf2 = brightfield_cy2.shape
x_max = max(x_cf1, x_cf2)
y_max = max(y_cf1, y_cf2)


brightfield_cy1_padded = np.pad(brightfield_cy1, pad_width=((y_max-y_cf1, 0), (0, x_max-x_cf1)), mode = 'median')
brightfield_cy2_padded = np.pad(brightfield_cy2, pad_width=((y_max-y_cf2, 0), (0, x_max-x_cf2)), mode = 'median')

brightfield_cy1_reduced = cv2.resize(brightfield_cy1_padded, (3000, 3000)) # 96.5% smaller = faster
brightfield_cy2_reduced = cv2.resize(brightfield_cy2, (3000, 3000)) # 96.5% smaller = faster


bf_cy1_processed = blur_noise_reduction(np.invert(brightfield_cy1_reduced))
bf_cy1_thresholded = threshold(bf_cy1_processed, 97)


bf_cy2_processed = blur_noise_reduction(np.invert(brightfield_cy2_reduced))
bf_cy2_thresholded = threshold(bf_cy2_processed, 97)




names = ["Cycle 1 (rd'd + proccessed)", "Cycle 2 (rd'd + proccessed)", "cycle2 warped"]

from scipy import ndimage
com_bf_cy1 = ndimage.center_of_mass(bf_cy1_thresholded)
com_bf_cy2 = ndimage.center_of_mass(bf_cy2_thresholded)

x_diff, y_diff = (com_bf_cy2[0] - com_bf_cy1[0], com_bf_cy2[1] - com_bf_cy1[1])

print(com_bf_cy1, com_bf_cy2)
print(x_diff, y_diff)

translation_matrix = np.float32([ [1,0,-y_diff], [0,1,-x_diff] ])

img_translation = cv2.warpAffine(bf_cy2_processed.copy(), translation_matrix, (3000,3000))

com_bf_cy3 = ndimage.center_of_mass(img_translation)
print(com_bf_cy3)

im = cv2.circle(bf_cy1_processed.copy(), (round(com_bf_cy1[0]), round(com_bf_cy1[1])), 20, (255, 255, 0) , -1)
im2 = cv2.circle(bf_cy2_processed.copy(), (round(com_bf_cy2[0]), round(com_bf_cy2[1])), 20, (255, 200, 0) , -1)
im2 = cv2.circle(im2, (round(com_bf_cy1[0]), round(com_bf_cy1[1])), 20, (100, 100, 0) , -1)
im3 = cv2.circle(img_translation.copy(), (round(com_bf_cy3[0]), round(com_bf_cy3[1])), 20, (255, 200, 0) , -1)

imgs = [im, im2, im3]


from matplotlib import pyplot as plt 

rows = len(imgs)

fig = plt.figure(figsize=(8*rows, 15*rows)) 

for i, img in enumerate(imgs):
    fig.add_subplot(rows, rows, i+1) 

    plt.imshow(img) 
    plt.axis('off') 
    plt.title(names[i]) 
    
mp, sizes, im = get_center_points(bf_cy1_thresholded.copy(), min_size=20)
mp2, sizes2, im2 = get_center_points(bf_cy2_thresholded.copy(), min_size=20)

mp2_array = np.array(mp2)

print(mp2_array)

x_rounded = round(x_diff)
y_rounded = round(y_diff)

print(com_bf_cy3)
for cell in mp2_array:
    cell[0] = cell[0] + x_rounded
    cell[1] = cell[1] + y_rounded

print(mp2_array)

# c(draw_gray_t(im, mp2_array))

c(im2)


current_max2 = 0
pt2 = []
for i, point in enumerate(mp2):
    if np.sum(point) > current_max2:
        current_max2 = np.sum(point)
        pt2 = point
        
print(current_max2, pt2)

current_min2 = 99999999
ptmin2 = []
for i, point2 in enumerate(mp2):
    if np.sum(point2) < current_min2:
        current_min2 = np.sum(point2)
        ptmin2 = point2
        
print(current_min2, ptmin2)




cf1_lr = lower_right(mp)
cf1_ul = upper_left(mp)
cf1_center = np.average([cf1_lr, cf1_ul],  axis = 0)
print("original cf1 ", cf1_lr, cf1_ul)

# print(cf1_center)
# image = cv2.line(im2.copy(), cf1_lr, cf1_ul, 120, 4) 
# c(image)

cf2_lr = lower_right(mp2)
cf2_ul = upper_left(mp2)
cf2_center = np.average([cf2_lr, cf2_ul],  axis = 0)
print("original cf2 ", cf2_lr, cf2_ul)

translation_cf2_to_cf1 = [- cf2_center[0] + cf1_center[0], - cf2_center[1] + cf1_center[1]]
# print(translation_cf2_to_cf1)

print("\nTRANSLATION NEEDED", translation_cf2_to_cf1, "\n")

translated_cf2_lr = [cf2_lr[0] + translation_cf2_to_cf1[0], cf2_lr[1] + translation_cf2_to_cf1[1]]
translated_cf2_ul = [cf2_ul[0] + translation_cf2_to_cf1[0], cf2_ul[1] + translation_cf2_to_cf1[1]]
t_cf2_center = np.average([translated_cf2_lr, translated_cf2_ul],  axis = 0)

print("translated c2", translated_cf2_lr, translated_cf2_ul)
print("and new center is", t_cf2_center)
# print(t_cf2_center)
# image2 = cv2.line(im2.copy(), cf_2lr, cf_2ul, 120, 4) 
# c(image2)


image = cv2.line(im.copy(), cf1_lr, cf1_ul, 120, 4) 
c(image)


import math

def slope(x1, y1, x2, y2): # Line slope given two points:
    return (y2-y1)/(x2-x1)

def angle(s1, s2): 
    return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))

lineA = (t_cf2_center, translated_cf2_ul)
lineB = (cf1_center, cf1_ul)

print(lineA)
print(lineB)

slope1 = slope(lineA[0][0], lineA[0][1], lineA[1][0], lineA[1][1])
slope2 = slope(lineB[0][0], lineB[0][1], lineB[1][0], lineB[1][1])

ang = angle(slope1, slope2)
print('Angle in degrees = ', ang)

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    
    # degrees = angle

    angle = angle*(math.pi/180)

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy




c(im2) # original 


## FIRST, SHIFT
cp = im2.copy()
### Doing shifting 
print(translation_cf2_to_cf1)
translation_matrix = np.float32([ [1,0,translation_cf2_to_cf1[0]], [0,1,translation_cf2_to_cf1[1]] ])
img_translation = cv2.warpAffine(cp.copy(), translation_matrix, (3000,3000))

c(img_translation)


## THEN, ROTATE
warp_dst = img_translation

center = t_cf2_center
angle = - (ang + .1)
scale = 1
rot_mat = cv2.getRotationMatrix2D( center, angle, scale )

warp_rotate_dst = cv2.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))

# cp = warp_rotate_dst

c(warp_rotate_dst)

