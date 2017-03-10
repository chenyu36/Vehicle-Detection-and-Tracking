import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
%matplotlib inline

# unpack the pickle file to restore parameters for function find_cars()
dist_pickle = pickle.load(open(feature_vector_pickle_file, "rb" ) )
svc = dist_pickle["svc"]
color_space = dist_pickle["color_space"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

images = glob.glob('./test_images/test*.jpg')

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, region_of_interest, 
              svc, X_scaler, orient, 
              pix_per_cell, cell_per_block, spatial_size, hist_bins, 
              vis=False, color_space = 'RGB2YCrCb', window_count=False):
    
    ystart = region_of_interest[0]
    ystop = region_of_interest[1]
    xstart = region_of_interest[2]
    xstop = region_of_interest[3]
    scale = region_of_interest[4]
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop, xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch, conv=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    bbox = [] # bounding box
    n_windows = 0
    for xb in range(nxsteps):
        for yb in range(nysteps):
            n_windows += 1
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)+xstart# add "xstart" as an offset
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                # only draw rectangle if visulization is required
                if vis is True:
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                bbox.append([(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)])
    
    if (window_count is True):
        return draw_img, bbox, n_windows
    else:
        return draw_img, bbox

# region definition: (ystart, ystop, xstart, xstop, scale)
region0 = (400, 646, 442, 1280, 1.2)
region1 = (500, 600, 300, 1280, 1.56)
region2 = (600, 646, 0, 1280, 2.0)
region3 = (240, 700, 0, 1280, 3.0)
region_of_interests = []
region_of_interests.append(region0)
region_of_interests.append(region1) 
region_of_interests.append(region2)
region_of_interests.append(region3)
color_conversion = 'RGB2'+ color_space
bbox_list = []
for file in images:
    img = mpimg.imread(file)
    out_img = img
    for i in range(1):
        out_img, bbox, n_windows = find_cars(out_img, region_of_interests[i], 
                        svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial_size, hist_bins, 
                        vis=True, color_space=color_conversion, window_count=True)      
#         if (file=='./test_images/test6.jpg' and  i == 0):
        bbox_list.append(bbox)
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=25)
    ax2.imshow(out_img)
    ax2.set_title('Labeled Image', fontsize=25)
print('number of sliding windows {}'.format(n_windows))  