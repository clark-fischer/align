
from PIL import Image
import numpy as np
import cv2
import time
import sys
import os
def do_(t):
    import time
    time.sleep(t)
    return t
    

def c(img):
    return Image.fromarray(img)

def alignImages(im1, im2):
  # Convert images to grayscale
  #im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  #im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
 
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(10000)
  keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2, None)
 
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = list(matcher.match(descriptors1, descriptors2, None))
 
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  print("Count of Matches:", len(matches))
  numGoodMatches = int(len(matches) * .01)
  print("Count of Good Matches:", numGoodMatches)
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
  height, width = np.array(im2.shape) * 1.5
  im1Reg = cv2.warpPerspective(im1, h, (int(width), int(height)))
 
  return im1Reg, h

def get_overlain(im1, im2, x, y):
    # print("images are", im1, im2)
    try:
                
        alngd, h = (alignImages(im1, im2))
        arr = np.array((c(alngd)).convert("RGBA"))
        arr[:, :, 3] = (255 * (arr[:, :, :3] != 0).any(axis=2)).astype(np.uint8)

        
        return ((x, y) , c(arr))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)







# OLD CODE
# x,y = block_points[1][2].astype(int)
# print(x,y)
# girth = int(block_points[0][0][0])
# print(girth)
# # print(x - girth, x+girth, y-girth, y+girth )    
# # print(slice(x-girth, y+girth), slice(x-girth, y+girth))
# cv2.rectangle(lol, (x-girth, y+girth), (x+girth, y-girth), (255, 255, 255), 102) 
# print((x-girth, y+girth), (x+girth, y-girth))
# c(lol)
# # LETS GOOOOOOO FIGURED IT OUUUTTTTT!!
# c(brightfield_cy1[y-girth:y+girth, x- girth:x+girth,])


# new_crew = c(np.zeros((SIZE_OF_ANALYSIS,SIZE_OF_ANALYSIS))).convert("RGBA")
# for i in block_points:
#     for j in i:
#         x,y = j.astype(int)
        
#         try:
#             alngd, h = (alignImages(brightfield_cy1[y-girth:y+girth, x- girth:x+girth,], brightfield_cy2[y-girth:y+girth, x- girth:x+girth,]))
#             arr = np.array((c(alngd)).convert("RGBA"))
#             arr[:, :, 3] = (255 * (arr[:, :, :3] != 0).any(axis=2)).astype(np.uint8)
#             print(x, y )
            
#             new_crew.paste(c(arr),(x-girth, y-girth), c(arr))
#             # new_crew[y-girth:y+girth, x- girth:x+girth,] = alngd
#         except Exception as e:
#             print(e)

# # new_crew.save("nc.png")

# # new_crew

