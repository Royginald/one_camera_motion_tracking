# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:13:03 2023

@author: RoyMc
"""

import numpy as np
import cv2

# --------------------- Functions ---------------------

def color_filter(image, desired_color, tol):
    # find how different each pixel is from the desired color
    img_float = np.array(image, dtype="float")
    
    mag_B = (img_float[:, :, 0] - desired_color[0])**2
    mag_G = (img_float[:, :, 1] - desired_color[1])**2
    mag_R = (img_float[:, :, 2] - desired_color[2])**2

    mag = (mag_R + mag_G + mag_B)**0.5;
    
    # if mangitude of difference is below tolerance, pixel becomes black, otherwise becomes white
    return np.array(255 * (np.arctan( (mag - tol)/1)/np.pi + 1/2 ), dtype="uint8")

def find_circle(image, x_search, y_search, spiral_spacing):
    found_pixel_x = [];
    found_pixel_y = [];
    bounds = np.zeros(4);
    height = len(image);
    width = len(image[0]);
    x_span = np.arange(width);
    y_span = np.arange(height);

    for i in range(len(x_search)):
        if image[int(y_search[i] + height/2)][int(x_search[i] + width/2)] < 15:
            found_pixel_x.append(int(x_search[i] + width/2));
            found_pixel_y.append(int(y_search[i] + height/2));
            
    # if at lease one black pixel is found, make outer bounds of black pixels
    if len(found_pixel_x) > 1:
        bounds[0] = max([min(found_pixel_x) - spiral_spacing*2, 0]);
        bounds[1] = min([max(found_pixel_x) + spiral_spacing*2, width-1]);
        bounds[2] = max([min(found_pixel_y) - spiral_spacing*2, 0]);
        bounds[3] = min([max(found_pixel_y) + spiral_spacing*2, height-1]);
        bounds = np.array(bounds, dtype="int")
    
        # cv2.rectangle(display_frame, (bounds[0], bounds[2]), (bounds[1], bounds[3]), display_color)
        
        # Normalize the pixels to be between 0 and 1, then invert image
        bits = image[bounds[2]:bounds[3], bounds[0]:bounds[1]]/255 - 1;

        # Use weiighted average to find the center of the black pixels
        x_center = sum(sum(bits*x_span[:(bounds[1]-bounds[0])]));
        y_center = sum(sum(bits.transpose()*y_span[:(bounds[3]-bounds[2])]));
        tot = sum(sum(bits));
                
        y_c = y_center/tot + bounds[2];
        x_c = x_center/tot + bounds[0];
        
        # Find the radius based on the area of black pixels
        R = np.sqrt(abs(tot)/np.pi);
    else:
        x_c = 0;
        y_c = 0;
        R = float("NaN")
                    
    return [x_c, y_c, R]

def find_coordiantes(x_center_px, y_center_px, rad_mm, rad_px, img_height_px, img_width_px, sensor_height_mm, sensor_width_mm, focal_length): 
    x_o = img_width_px*0.5 - x_center_px;
    y_o = img_height_px*0.5 - y_center_px;
    
    z =  focal_length * rad_mm * img_height_px / ( rad_px * sensor_height_mm );
    y =  y_o * z * sensor_width_mm  / ( focal_length * img_width_px );
    x = -x_o * z * sensor_height_mm / ( focal_length * img_height_px );
    
    return [x, y, z]

# ------------------- Inital Set up -------------------

# user changable parameters
width = 1280; # pixels
height = 720; # pixels

# width = 1960; # pixels
# height = 1080; # pixels

spiral_spacing = 15;
real_radius = 66/2; # mm
focal_length = 3.67; # mm
sensor_height = 2.72; # mm
sensor_width = 4.836; # mm

display_color = (255, 0, 255);
desired_color = [202, 45, 2]; # blue
# desired_color = [20, 4, 230]; # BGR, Red
# desired_color= [112, 175, 109]; # BGR, Green

tol = 130;

# non-user changable parameters

spiral_spacing_x = width/max(width, height) * spiral_spacing;
spiral_spacing_y = height/max(width, height) * spiral_spacing;

x_span = np.arange(width);
y_span = np.arange(height);

bounds = np.zeros(4);

# setting up search indexes
x = [];
y = [];
x_current = 0;
y_current = 0;

for i in range(int(2*width/spiral_spacing_x-1)): # theta
    x_dir = np.cos(i*np.pi/2);
    y_dir = np.sin(i*np.pi/2);
    for j in range(1, int(np.floor(i/2))):
        x.append(int(x_current));
        y.append(int(y_current));
        x_current += spiral_spacing_x*x_dir;
        y_current += spiral_spacing_y*y_dir;
        
# reading from a webcamera
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW);
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width);
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height);

# ------------------- Infinite loop -------------------

while True:
    ret, frame = video_capture.read(); # NOTE: frame is y, x, colorspace is BGR
    
    filtered = color_filter(frame, desired_color, tol);
    [x_c, y_c, R] = find_circle(filtered, x, y, spiral_spacing);
    [x_r, y_r, z_r] = find_coordiantes(x_c, y_c, real_radius, R, height, width, sensor_height, sensor_width, focal_length)
    
    if x_c > 0 and y_c > 0 and ~np.isnan(R):
        cv2.circle(frame, (round(x_c), round(y_c)), round(R), display_color, 1);
        cv2.rectangle(frame, (0, 0), (480, 55), (0, 0, 0), -1)
        cv2.putText(frame, "x: %.1f y: %.1f z: %.1f" % (x_r, y_r, z_r), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    
    cv2.imshow("Window", frame);
    
    #This breaks on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
    
video_capture.release();
cv2.destroyAllWindows();


