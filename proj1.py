import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
# image = mpimg.imread('test_images/solidWhiteCurve.jpg')
# image = mpimg.imread('test_images/solidYellowCurve2.jpg')
# image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
# plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    sigma=0.10
    v = np.median(img)
    # print("median:",v)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)

    # return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices,ignore_mask_color=None):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    mask_inner = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if not ignore_mask_color:
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255            
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=RED, thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def HSL_filtered(image):
    hsl_image = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(hsl_image, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(hsl_image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

# def HSV_filtered(image):
#     hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
#     # define range of blue color in HSV
#     lower_yellow = np.array([58,13,60])
#     upper_yellow = np.array([64,98,100])
#     # Threshold the HSV image to get only blue colors
#     mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

#     lower_white = np.array([0,0,60])
#     upper_white = np.array([359,3,100])
#     # Threshold the HSV image to get only blue colors
#     mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

#     mask = cv2.bitwise_or(mask_white, mask_yellow)
#     # Bitwise-AND mask and original image
#     res = cv2.bitwise_and(image,image, mask= mask)

import os
os.listdir("test_images/")
if not os.path.isdir("test_videos_output"):
    print("No dir")
    os.mkdir("test_videos_output")

def draw_lane_lines(image,):
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 20 #minimum number of pixels making up a line
    max_line_gap = 300    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    # hsl_image = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    hsl_image = HSL_filtered(image)
    # hsl_image = HSV_filtered(image)
    gray_image = grayscale(hsl_image)
    gaussian_image = gaussian_blur(gray_image,15)

    imgX = image.shape[1]
    imgY = image.shape[0]

    left_lower = (imgX*0.10,imgY)
    left_upper = (imgX*0.42,imgY*0.60)
    right_upper = (imgX*0.57,imgY*0.60)
    right_lower = (imgX*0.92,imgY)

    left_lower_in = (imgX*0.20,imgY)
    left_upper_in = (imgX*0.50,imgY*0.62)
    right_upper_in = (imgX*0.50,imgY*0.62)
    right_lower_in = (imgX*0.80,imgY)

    vertices = np.array([[left_lower,left_upper,right_upper,right_lower]], dtype=np.int32)
    vertices_inner = np.array([[left_lower_in,left_upper_in,right_upper_in,right_lower_in]], dtype=np.int32)
    
    edges = canny(gaussian_image,95,100)
    masked_image = region_of_interest(edges, vertices)
    
    # masked_image = region_of_interest(gaussian_image, vertices,vertices_inner)
    # masked_image = gaussian_blur(masked_image,11)
    # edges = canny(masked_image,95,100)
    # edges = region_of_interest(edges, vertices,vertices_inner)
   

    plt.figure()
    plt.imshow(edges)
    plt.figure()
    plt.imshow(masked_image)
    plt.title("masked_image")
    # line_image = hough_lines(inner_mask, rho, theta, threshold, min_line_len, max_line_gap)

    color_edges = np.dstack((image[:,:,0], image[:,:,1], image[:,:,2])) 
    # color_edges = np.dstack((masked_image, masked_image, masked_image)) 
    # color_edges = np.dstack((edges, edges, edges)) 

    # line interpolation
    def interpolate_lines(line_image,_color=RED,line_thickness=15):
        imY,imX = line_image.shape
        # print("imx/2",imX/2,imY)
        lines = cv2.HoughLinesP(line_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        
        LeftSet = []
        RightSet = []
        for line in lines:
            p1 = (line[0][0],line[0][1])
            p2 = (line[0][2],line[0][3])
            if p1[0] < imX/2:
                LeftSet.append(p1)
            else:
                RightSet.append(p1)
            if p2[0] < imX/2:
                LeftSet.append(p2)
            else:
                RightSet.append(p2) 
        LeftSet = np.array(LeftSet)
        RightSet = np.array(RightSet)

        LeftX = np.array([pt[0] for pt in LeftSet])
        LeftY = np.array([pt[1] for pt in LeftSet])
                
        RightX = np.array([pt[0] for pt in RightSet])
        RightY = np.array([pt[1] for pt in RightSet])
        
        #------------- Slope Approach ------------------#
        # slope = lambda p1,p2: abs((p2[1]-p1[1])/(p2[0]-p1[0]+1e-10))
        
        # def slope(p1,p2):
        #     if(p2[1] > p1[1]):
        #         _slope = (p2[1]-p1[1])/(p2[0]-p1[0]+1e-10)
        #     else:    
        #         _slope = (p1[1]-p2[1])/(p1[0]-p2[0]+1e-10)
            
        #     _intercept = p1[1]-_slope*p1[0]
        #     return _slope,_intercept    

        # LeftSlope = [ slope(LeftSet[idx],LeftSet[idx+1])[0] for idx in range(len(LeftSet)-1)]
        # LeftIntercept = [ slope(LeftSet[idx],LeftSet[idx+1])[1] for idx in range(len(LeftSet)-1)]
        # LeftSlope = np.median(np.array(LeftSlope))
        # LeftIntercept = np.median(np.array(LeftIntercept))
        # # print("LeftSlope",LeftSlope)

        # RightSlope = [ slope(RightSet[idx],RightSet[idx+1])[0] for idx in range(len(RightSet)-1)]
        # RightIntercept = [ slope(RightSet[idx],RightSet[idx+1])[1] for idx in range(len(RightSet)-1)]
        # RightSlope = np.median(np.array(RightSlope))
        # RightIntercept = np.median(np.array(RightIntercept)) 
        
        # print("RightSlope",RightSlope)
        #------------- Slope Approach ------------------#

        # curve = lambda y,a: int(a[0]*y+a[1])
        curve = lambda y,a: int((y-a[1])/a[0])
        
        a = np.polyfit(LeftX, LeftY, 1)
        b = np.polyfit(RightX, RightY, 1)
        
        LeftY.sort()
        RightY.sort()
        # print(a)        
        # print(b)        

        # a = [LeftSlope,LeftIntercept]
        # b = [RightSlope,RightIntercept]

        LeftPts = [(curve(_y,a),_y) for _y in LeftY]
        RightPts = [(curve(_y,b),_y) for _y in RightY]

        LeftPts = np.append(LeftPts,np.array([[curve(imY,a),imY]]),axis=0)
        RightPts = np.append(RightPts,np.array([[curve(imY,b),imY]]),axis=0)

        # LeftSet = np.array(LeftSet)
        # RightSet = np.array(RightSet)
        
        new_line_img = np.zeros((line_image.shape[0], line_image.shape[1], 3), dtype=np.uint8)

        cv2.polylines(new_line_img, np.int32([LeftPts]), 0, _color,line_thickness)
        cv2.polylines(new_line_img, np.int32([RightPts]), 0, _color,line_thickness)
        return new_line_img

    line_image = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap)
    # new_line_img = interpolate_lines(line_image[:,:,0])
    new_line_img = interpolate_lines(masked_image)
    lines_edges = weighted_img(new_line_img,color_edges)
    # return edges
    return lines_edges
result = draw_lane_lines(image)
plt.figure()
plt.imshow(result)
plt.show()

from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    try:
        result = draw_lane_lines(image)
    except Exception as e:
        result = image
    return result

'''
white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
'''
'''
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)
'''


challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
