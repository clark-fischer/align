
from PIL import Image
import numpy as np
import diplib
from PIL import ImageFilter
import astroalign as aa
import cv2
import sys
import math

sys.setrecursionlimit(53000) # override needed for computing midpoints, which uses a recursive function
Image.MAX_IMAGE_PIXELS = 366498276 # override is needed, or else it gives a DecompressionBombError


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

def slope(x1, y1, x2, y2): # Line slope given two points:
    return (y2-y1)/(x2-x1)

def angle(s1, s2): 
    return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))

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



layer_to_align = 2

img_cy1 = Image.open('originals/ImageSLIDE4-CYCLE1.tif')
img_cy1.seek(layer_to_align) # navigate to brightfield
brightfield_cy1 = np.array(img_cy1)
brightfield_cy1 = (brightfield_cy1/256).astype('uint8') # don't need the whole range of values, so this reduces mem size, + improves access speed

img_cy2 = Image.open('originals/ImageARRAY4-CYCLE2.tif')
img_cy2.seek(layer_to_align) # navigate to brightfield
brightfield_cy2 = np.array(img_cy2)
brightfield_cy2 = (brightfield_cy2/256).astype('uint8') 

# fix sizing
y_cf1, x_cf1 = brightfield_cy1.shape
y_cf2, x_cf2 = brightfield_cy2.shape
max_dim = max(x_cf1, x_cf2, y_cf1, y_cf2)

scale_down_factor = 5
new_size = max_dim // scale_down_factor

print(brightfield_cy1.shape, brightfield_cy2.shape)

brightfield_cy1_padded = np.pad(brightfield_cy1, pad_width=((max_dim-y_cf1, 0), (0, max_dim-x_cf1)), mode = 'median')
brightfield_cy2_padded = np.pad(brightfield_cy2, pad_width=((max_dim-y_cf2, 0), (0, max_dim-x_cf2)), mode = 'median')

brightfield_cy1_reduced = cv2.resize(brightfield_cy1_padded, (new_size, new_size)) # 96.5% smaller = faster
brightfield_cy2_reduced = cv2.resize(brightfield_cy2_padded, (new_size, new_size)) # 96.5% smaller = faster


### clean up
img_cy1 = None
img_cy2 = None
brightfield_cy1 = None
brightfield_cy2 = None
brightfield_cy1_padded = None
brightfield_cy2_padded = None

bf_cy1_processed = blur_noise_reduction(np.invert(brightfield_cy1_reduced))
bf_cy1_thresholded = threshold(bf_cy1_processed, 97)

bf_cy2_processed = blur_noise_reduction(np.invert(brightfield_cy2_reduced))
bf_cy2_thresholded = threshold(bf_cy2_processed, 97)

mp, sizes, im = get_center_points(bf_cy1_thresholded.copy(), min_size=20)
mp2, sizes2, im2 = get_center_points(bf_cy2_thresholded.copy(), min_size=20)

mp2_array = np.array(mp2)
x_rounded = round(x_diff)
y_rounded = round(y_diff)

for cell in mp2_array:
    cell[0] = cell[0] + x_rounded
    cell[1] = cell[1] + y_rounded


cf1_lr = lower_right(mp)
cf1_ul = upper_left(mp)
cf1_center = np.average([cf1_lr, cf1_ul],  axis = 0)
print("original cf1 ", cf1_lr, cf1_ul)

cf2_lr = lower_right(mp2)
cf2_ul = upper_left(mp2)
cf2_center = np.average([cf2_lr, cf2_ul],  axis = 0)
print("original cf2 ", cf2_lr, cf2_ul)

translation_cf2_to_cf1 = [- cf2_center[0] + cf1_center[0], - cf2_center[1] + cf1_center[1]]

print("\nTRANSLATION NEEDED", translation_cf2_to_cf1, "\n")

translated_cf2_lr = [cf2_lr[0] + translation_cf2_to_cf1[0], cf2_lr[1] + translation_cf2_to_cf1[1]]
translated_cf2_ul = [cf2_ul[0] + translation_cf2_to_cf1[0], cf2_ul[1] + translation_cf2_to_cf1[1]]
t_cf2_center = np.average([translated_cf2_lr, translated_cf2_ul],  axis = 0)

print("translated c2", translated_cf2_lr, translated_cf2_ul)
print("and new center is", t_cf2_center)

image = cv2.line(im.copy(), cf1_lr, cf1_ul, 120, 4) 

lineA = (t_cf2_center, translated_cf2_ul)
lineB = (cf1_center, cf1_ul)

slope1 = slope(lineA[0][0], lineA[0][1], lineA[1][0], lineA[1][1])
slope2 = slope(lineB[0][0], lineB[0][1], lineB[1][0], lineB[1][1])

ang = angle(slope1, slope2)
print('Angle in degrees = ', ang)

## FIRST, SHIFT
cp = im2.copy() 
print(translation_cf2_to_cf1)
translation_matrix = np.float32([ [1,0,translation_cf2_to_cf1[0]], [0,1,translation_cf2_to_cf1[1]] ])
img_translation = cv2.warpAffine(cp.copy(), translation_matrix, (new_size,new_size))

print("shift success!")
# c(img_translation)


## THEN, ROTATE
warp_dst = img_translation

center = t_cf2_center
angle = - (ang + .1)
scale = 1
rot_mat = cv2.getRotationMatrix2D( center, angle, scale )

warp_rotate_dst = cv2.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))

# cp = warp_rotate_dst

c(warp_rotate_dst)


# cp = brightfield_cy1
# ### Doing shifting 
# print(translation_cf2_to_cf1)
# translation_matrix = np.float32([ [1,0,translation_cf2_to_cf1[0]], [0,1,translation_cf2_to_cf1[1]] ])
# img_translation = cv2.warpAffine(cp.copy(), translation_matrix, (170000,17000))




# okay, now to apply to real image, we need to scale up the translation + the center of rotation


upscaled_translation = np.array(translation_cf2_to_cf1) * scale_down_factor
upscaled_center = t_cf2_center * scale_down_factor

print("translate from", translation_cf2_to_cf1, "to", upscaled_translation)
print("rotational center from", t_cf2_center, "to", upscaled_center)

###### don't change!
img_cy2 = Image.open('originals/ImageARRAY4-CYCLE2.tif')
img_cy2.seek(7) # navigate to brightfield
cp = np.array(img_cy2)
print("before padding", cp.shape)
cp = np.pad(cp, pad_width=((max_dim-y_cf2, 0), (0, max_dim-x_cf2)), mode = 'median')
print("after padding ", cp.shape)
######


# clip


new_size = (cp.shape[0])
cp = cv2.resize(cp, (new_size, new_size)) # 96.5% smaller = faster

# translating
translation_matrix = np.float32([ [1,0,translation_cf2_to_cf1[0] * 5], [0,1,translation_cf2_to_cf1[1] * 5] ])
img_translation = cv2.warpAffine(cp, translation_matrix, cp.shape)

# rotating
rot_mat = cv2.getRotationMatrix2D( t_cf2_center * 5, -(ang + .1), 1 )
warp_rotate_dst = cv2.warpAffine(img_translation, rot_mat, (img_translation.shape[1], img_translation.shape[0]))



### also don't forget the padded cycle1!
img_cy1 = Image.open('originals/ImageSLIDE4-CYCLE1.tif')
img_cy1.seek(7) # navigate to brightfield
cp = np.array(img_cy1)
print("before padding", cp.shape)
cp = np.pad(cp, pad_width=((max_dim-y_cf1, 0), (0, max_dim-x_cf1)), mode = 'median')
print("after padding ", cp.shape)



#######
#######
#######
#######
#######
####### 



layer_to_align = 7

img_cy1 = cp
brightfield_cy1 = np.array(img_cy1)
brightfield_cy1 = (brightfield_cy1/256).astype('uint8') # don't need the whole range of values, so this reduces mem size, + improves access speed

img_cy2 = warp_rotate_dst
brightfield_cy2 = np.array(img_cy2)
brightfield_cy2 = (brightfield_cy2/256).astype('uint8') 

img_cy1 = None
img_cy2 = None


y, height = 0, 10000
x, width = 0, 10000

img1 = brightfield_cy1[y:height+y, x:width+x]
img2 = brightfield_cy2[y:height+y, x:width+x]



def alignImages(im1, im2):
  # Convert images to grayscale
  #im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  #im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
 
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(5000)
  keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2, None)
 
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = list(matcher.match(descriptors1, descriptors2, None))
 
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  print("Count of Matches:", len(matches))
  numGoodMatches = int(len(matches) * .05)
  print("Distance of worst match:", matches[numGoodMatches].distance)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  #imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  #cv2.imwrite("matches.jpg", imMatches)
 
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
 
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
 
  return im1Reg, h
 
num_alignments = 6
alignable = img1.copy()
for align_n in range(num_alignments):
  alignable, h = alignImages(alignable,img2)
  



# 0: "only neither is  solid", 
# 1: "only cycle 1 is solid",
# 2: "only cycle 2 is solid", 
# 3: "both are solid",     




c(threshold(adjust_contrast(alignable), 98))



def color_in(mat_orig, min_size=0):
    mat = (mat_orig)
    gray_edit_copy = mat_orig.copy()

    def fill_blob(x, y):
        
        mat[y][x] = False
        gray_edit_copy[y][x] = False
        
        #  right
        if  (x != mat.shape[1] - 1) and mat[y][x+1]:
            (fill_blob(x+1, y))
        # down
        if  (y != mat.shape[0] - 1) and mat[y+1][x]:
            (fill_blob(x, y+1))
        # left
        if  (x != 0) and mat[y][x-1]:
            (fill_blob(x-1, y))
        # up
        if  (y != 0) and  mat[y-1][x]:
            fill_blob(x, y-1)
        
    for y, line in enumerate(mat):
        for x, pixel in enumerate(line):
            if pixel == 3:
                fill_blob(x, y)
                
    
    return gray_edit_copy


im1_beads = threshold(adjust_contrast(im1_r_onto_im2_third4), 85)
im2_beads = (threshold(adjust_contrast(img2), 90))

im1_beads[im1_beads == 255] = 1
im2_beads[im2_beads == 255] = 2

overlaid = np.add(im1_beads, im2_beads)



flow = cv.calcOpticalFlowFarneback(np.array(im1_r_onto_im2_third4), 
                                   np.array(img2), 
                                   None, 0.5, 3, 15, 3, 5, 1.2, 0)

def put_optical_flow_arrows_on_image(image, optical_flow_image, threshold=2.0, skip_amount=30):
    # Don't affect original image
    image = image.copy()
    
    # Turn grayscale to rgb if needed
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=2)
    
    # Get start and end coordinates of the optical flow
    flow_start = np.stack(np.meshgrid(range(optical_flow_image.shape[1]), range(optical_flow_image.shape[0])), 2)
    flow_end = (optical_flow_image[flow_start[:,:,1],flow_start[:,:,0],:1]*3 + flow_start).astype(np.int32)
    

    # Threshold values
    norm = np.linalg.norm(flow_end - flow_start, axis=2)
    norm[norm < threshold] = 0
    
    # Draw all the nonzero values
    nz = np.nonzero(norm)
    for i in range(0, len(nz[0]), skip_amount):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(image,
                        pt1=tuple(flow_start[y,x]), 
                        pt2=tuple(flow_end[y,x]),
                        color=(0, 255, 255), 
                        thickness=1, 
                        tipLength=.2)
    return image

c(put_optical_flow_arrows_on_image(im1_r_onto_im2_third4,flow))


