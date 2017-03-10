import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from collections import deque


# Define a class to receive the characteristics of each heatmap derived centroid detection
class Vehicle():
    def __init__(self):
        # was the centroid detected in the last iteration?
        self.detected = False
        # number of times the vehicle has been detected
        self.n_detection = 0
        # center value of the centroid
        self.center = None
        # width of the detected centroid
        self.width = None
        # height of the detected centroid
        self.height = None
        # centers of the last n frames of the centroid
        self.recent_centers = []
        self.recent_w = []
        self.recent_h = []
        # average center over the last n frames
        self.avg_center = None
        # average width of the centroid over the last n frames
        self.avg_width = None
        # average height of the centroid over the last n frames
        self.avg_height = None
  
 
car = Vehicle()



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # print('box is {}'.format(box))
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    centroid_list = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 5)
        centroid_list.append(bbox)
    # Return the image and bbox of the detected vehicles 
    return img, centroid_list

def draw_labeled_bboxes_with_filter(img, labels, thres1=(30, 54), thres2=(80, 70), y_min=510):
    centroid_list = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image if its size is above the threshold
        width = np.max(nonzerox) - np.min(nonzerox)
        height = np.max(nonzeroy) - np.min(nonzeroy)
        # small/narrow box that are located near the bottom of the screen is not representative of a car
        # box with very narrow width is not representative of a car
        # box with certain size while plausible in distance is not representative of a car when near
        if not (((width < thres1[0] or height < thres1[1]) and np.min(nonzeroy) > y_min) or (width < thres1[0]) or
                 width < thres2[0] and height < thres2[1] and np.min(nonzeroy) > y_min):
            cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 5)
            centroid_list.append(bbox)
    # Return the image and bbox of the detected vehicles 
    return img, centroid_list

def draw_avg_centroid_bboxes(img, centroid, index):
    if (centroid.avg_center is not None):      
        # Define a bounding box based on min/max x and y
        min_x = (centroid.avg_center[0] - centroid.avg_width//2).astype(np.int)
        min_y = (centroid.avg_center[1] - centroid.avg_height//2).astype(np.int)
        max_x = (centroid.avg_center[0] + centroid.avg_width//2).astype(np.int)
        max_y = (centroid.avg_center[1] + centroid.avg_height//2).astype(np.int)    
        bbox = ((min_x, min_y), (max_x, max_y))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 3)
        # draw car index in the center of the bbox
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(index),(centroid.avg_center[0],centroid.avg_center[1]), font, 1,(0,255,0),2,cv2.LINE_AA)
#         print('drawing average bbox')
#         print('bbox[0] is {}'.format(bbox[0]))
#         print('bbox[1] is {}'.format(bbox[1]))          
        # Return the image
    return img

def display_text(img, dbg_string):
#     print('type of image is {}'.format(type(img)))
#     print('image is {}'.format(img))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, dbg_string,(600,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    
def get_window_center(window):
    x_min = window[0][0]
    y_min = window[0][1]
    x_max = window[1][0]
    y_max = window[1][1]
    center = ((x_min+x_max)//2, (y_min+y_max)//2)
    return center


def average_single_centroid(window, car, n_frame = 11, update_thres = 0.15, img=None):
    centroid_dim = []

    if window is not None:
        x_min = window[0][0]
        y_min = window[0][1]
        x_max = window[1][0]
        y_max = window[1][1]
        center = ((x_min+x_max)//2, (y_min+y_max)//2)
        w = x_max - x_min
        h = y_max - y_min
        data = (center, w, h)
        centroid_dim.append(data)
        
        # make sure there are enough frames to average
        if (car.n_detection < n_frame):
            car.n_detection = car.n_detection+1         
            # capture data
            car.detected = True
            # update only if it's below the update threshold (as percentage)
            # center
            if (len(car.recent_centers) > 0 or 
                len(car.recent_w) > 0 or 
                len(car.recent_h) > 0):
                delta1 = abs(car.recent_centers[-1][0] - center[0])/car.recent_h[-1]  
                delta2 = abs(car.recent_w[-1] - w)/car.recent_w[-1] 
                delta3 = abs(car.recent_h[-1] - h)/car.recent_h[-1] 
                # only update if all delta are below threshold            
#                 if (delta1 < update_thres or delta2 < update_thres or delta3 < update_thres):
                if True:
                    # center
                    car.center = center
                    car.recent_centers.append(center)
                    # width
                    car.width = w
                    car.recent_w.append(w)  
                    # height
                    car.height = h
                    car.recent_h.append(h)                 
                    # print('update centroid data')
                    if img is not None:
                        display_text(img, 'update centroid data')
                else:
                    #print('no update to centroid data')
                    if img is not None:
                        display_text(img, 'no update to centroid data')     
            else:
                car.center = data[0]
                car.recent_centers.append(center)
                car.width = w
                car.recent_w.append(w)  
                car.height = h
                car.recent_h.append(h)                 
        else:
            # center
            delta1 = abs(car.recent_centers[-1][0] - center[0])/car.recent_h[-1] 
            #print('center delta x percent is {}'.format(delta1))  
            delta2 = abs(car.recent_w[-1] - w)/car.recent_w[-1] 
            delta3 = abs(car.recent_h[-1] - h)/car.recent_h[-1] 
            
            # only update if all delta are below threshold            
#             if (delta1 < update_thres or delta2 < update_thres or delta3 < update_thres):
            if True:
                # center
                car.recent_centers.pop(0)
                car.center = center
                car.recent_centers.append(center)
                # width
                car.recent_w.pop(0)
                car.width = w
                car.recent_w.append(w)  
                # height
                car.recent_h.pop(0)
                car.height = h
                car.recent_h.append(h)                 
                #print('update centroid data')
                if img is not None:
                    display_text(img, 'update centroid data')
            else:
                #print('no update to centroid data')
                if img is not None:
                    display_text(img, 'no update to centroid data')            
                              
            # take weighted average
            if n_frame == 5:
                weights = (-3., 12., 17., 12.,-3.)
            if n_frame == 9:
                weights = (-21., 14., 39., 54., 59., 54., 39., 14., -21)
            if n_frame == 11:
                weights = (-36., 9., 44., 69., 84., 89., 84., 69., 44., 9., -36)                  
            w_sum = np.sum(weights)
            weights = weights/w_sum
#             print('length of recent_centers {}'.format(len(car.recent_centers)))
            if (len(car.recent_centers) == n_frame):
                car.avg_center = np.average(car.recent_centers, axis=0, weights=weights).astype(np.int)
                car.avg_width = np.average(car.recent_w, axis=0, weights=weights).astype(np.int)
                car.avg_height = np.average(car.recent_h, axis=0, weights=weights).astype(np.int)
                
    return img, centroid_dim, car     

# Read in a test images
images = glob.glob('./test_images/test*.jpg')
test_images = []
for file in images:
    image = mpimg.imread(file)
    test_images.append(image)

for i in range(len(test_images)):
    heat = np.zeros_like(test_images[i][:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,bbox_list[i])

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,4)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img, centroid_list = draw_labeled_bboxes_with_filter(np.copy(test_images[i]), labels)
    n_of_cars =labels[1]

    print(n_of_cars, 'cars found')

    # Visualization
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(draw_img)
    ax1.set_title('Car Positions', fontsize=25)
    ax2.imshow(heatmap, cmap='hot')
    ax2.set_title('Heat Map', fontsize=25)
    filename = './output_images/car_positions_vs_heatmap' + str(i) + '.png'
    f.savefig(filename)