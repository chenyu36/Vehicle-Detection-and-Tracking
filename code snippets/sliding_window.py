import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

image = mpimg.imread('bbox-example-image.jpg')

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if (x_start_stop[0] is None and x_start_stop[1] is None):
        x_start_stop[0] = 0
        x_start_stop[1] = img.shape[1]
        y_start_stop[0] = 0
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    x_search_span = x_start_stop[1] - x_start_stop[0]
    y_search_span = y_start_stop[1] - y_start_stop[0]
    #print('x_search_span {}'.format(x_search_span))
    # Compute the number of pixels per step in x/y
    pixel_per_step_x = xy_window[0]*xy_overlap[0]
    pixel_per_step_y = xy_window[1]*xy_overlap[1]
    # print('pixel_per_step_x {}'.format(pixel_per_step_x))
    # Compute the number of windows in x/y
    n_window_x = math.floor((x_search_span - xy_window[0])/pixel_per_step_x) + 1
    n_window_y = math.floor((y_search_span - xy_window[1])/pixel_per_step_y) + 1
    print('n_window_x {}'.format(n_window_x))
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
        # Append window position to list
    for j in range(n_window_y):
        for i in range(n_window_x):
            top_left_x = int(x_start_stop[0] + i*pixel_per_step_x)
            top_left_y = int(y_start_stop[0] + j*pixel_per_step_y)
            bottom_right_x = top_left_x + xy_window[0]
            bottom_right_y = top_left_y + xy_window[1]
            top_left = (top_left_x, top_left_y)
            bottom_right = (bottom_right_x, bottom_right_y)
            window_list.append((top_left, bottom_right))
    # Return the list of windows
    return window_list

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
                       
window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)                    
plt.imshow(window_img)