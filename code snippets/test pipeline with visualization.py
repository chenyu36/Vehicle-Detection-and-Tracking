import itertools
region0 = (400, 646, 442, 1280, 1.2)
region1 = (500, 600, 300, 1280, 1.56)
region2 = (600, 646, 0, 1280, 2.0)
region3 = (240, 700, 0, 1280, 3.0)
region_of_interests = []
region_of_interests.append(region0)
region_of_interests.append(region1) 
region_of_interests.append(region2)
region_of_interests.append(region3)
detected_window_list = []
frame_counter = 0
frame_thres = 11
def test_pipeline(img):
    global frame_counter
    bbox_list = []
    # only use one window size to search
    for i in range(1):
        out_img, bbox = find_cars(img, region_of_interests[i], 
                                    svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        bbox_list =  bbox_list +  bbox  
    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat,bbox_list)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,3)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img, centroid_list = draw_labeled_bboxes(np.copy(img), labels)
    
    # Visualization of the first N frames
    if (frame_counter <= frame_thres):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(draw_img)
        title1 = 'Frame' + str(frame_counter)
        ax1.set_title(title1, fontsize=25)
        ax2.imshow(heatmap, cmap='hot')
        title2 = 'HeatMapOfFrame' + str(frame_counter)
        ax2.set_title(title2, fontsize=25)
        filename = './output_images/' + title2 + '.png'
        f.savefig(filename)        
    
    # store the windows detected in the last N cycles, where N is 11 in this case
    detected_window_list.append(centroid_list)
    # remove the oldest windows that was detected N cycles ago
    if len(detected_window_list) > frame_thres:
        detected_window_list.pop(0)
        heat_integrate = np.zeros_like(img[:,:,0]).astype(np.float)
        
        flatten = list(itertools.chain.from_iterable(detected_window_list))
        
        # Add heat to each box in flatten box list
        heat_integrate = add_heat(heat_integrate,flatten)
        
        # Apply threshold to help remove false positives
        heat_integrate = apply_threshold(heat_integrate,5)
        
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat_integrate, 0, 255)        
        
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        out_img, heatmap_list = draw_labeled_bboxes_with_filter(np.copy(img), labels)
        
        # Visualize the resulting bounding boxes with integrated heat map
        if (frame_counter == (frame_thres + 1)):
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.imshow(draw_img)
            title1 = 'ResultingBoundingBoxAtFrame' + str(frame_counter)
            ax1.set_title(title1, fontsize=25)
            ax2.imshow(heatmap, cmap='hot')
            title2 = 'IntegratedHeatMapAtFrame' + str(frame_counter)
            ax2.set_title(title2, fontsize=25)
            filename = './output_images/' + title2 + '.png'
            f.savefig(filename)   
    
    
    frame_counter += 1
    
    return out_img