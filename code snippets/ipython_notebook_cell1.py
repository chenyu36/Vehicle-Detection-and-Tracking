import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from pathlib import Path

#declar the name of the pickle file to save
feature_vector_pickle_file = './feature_vector_pickle_fullset.p'
feature_vector_file = Path(feature_vector_pickle_file)

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)    

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    print('Extracting features, this may take a while...')
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# Divide up into cars and notcars

# cars
car_filepath = []
car_filepath.append('../labeled dataset/project/vehicles/GTI_Far/*.png')
car_filepath.append('../labeled dataset/project/vehicles/GTI_Left/*.png')
car_filepath.append('../labeled dataset/project/vehicles/GTI_MiddleClose/*.png')
car_filepath.append('../labeled dataset/project/vehicles/GTI_Right/*.png')
car_filepath.append('../labeled dataset/project/vehicles/KITTI_extracted/*.png')
# not cars
notcar_filepath = []
notcar_filepath.append('./training_set/non-vehicles/GTI/*.png')
notcar_filepath.append('./training_set/non-vehicles/Extras/*.png')

car_images = []
notcar_images = []
for files in car_filepath:
    print('car files are {}'.format(files))
    car_images = car_images + glob.glob(files)
print('Number of car images in the data set {}'.format(len(car_images)))

for files in notcar_filepath:
    print('notcar files are {}'.format(files))
    notcar_images = notcar_images + glob.glob(files)
print('Number of non-car images in the data set {}'.format(len(notcar_images))) 

cars = []
notcars = []
for image in car_images:
        cars.append(image)
for image in notcar_images:
        notcars.append(image)        

# Note: During testing, reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
reduce_sample_size_for_testing = False
if reduce_sample_size_for_testing is True:
    sample_size = 500
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

# Apply these parameters to extract the features.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True # Spatial features on or off
hist_feat = True    # Histogram features on or off
hog_feat = True     # HOG features on or off


# check if we already created the feature vectors
if feature_vector_file.is_file():
    print('Feature vectors are already created')
    # load the feature vectors and corresponding parameters
    with open(feature_vector_pickle_file, mode='rb') as f:
        dist_pickle = pickle.load(f)
        svc = dist_pickle["svc"]
        color_space = dist_pickle["color_space"]
        X_scaler = dist_pickle["scaler"]
        orient = dist_pickle["orient"]
        pix_per_cell = dist_pickle["pix_per_cell"]
        cell_per_block = dist_pickle["cell_per_block"]
        spatial_size = dist_pickle["spatial_size"]
        hist_bins = dist_pickle["hist_bins"]
    print('Model trained using:',color_space,'color space |',orient,'orientations |',pix_per_cell,
        'pixels per cell |', cell_per_block,'cells per block |', hog_channel, 'hog channel')
else:
    t=time.time()
    car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',color_space,'color space |',orient,'orientations |',pix_per_cell,
        'pixels per cell |', cell_per_block,'cells per block |', hog_channel, 'hog channel')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    # save the result for later use    
    dist_pickle = {}
    dist_pickle["svc"] = svc
    dist_pickle["color_space"] = color_space    
    dist_pickle["scaler"] = X_scaler
    dist_pickle["orient"] = orient
    dist_pickle["pix_per_cell"] = pix_per_cell   
    dist_pickle["cell_per_block"] = cell_per_block
    dist_pickle["spatial_size"] = spatial_size
    dist_pickle["hist_bins"] = hist_bins   
    pickle.dump( dist_pickle, open( feature_vector_pickle_file, 'wb' ) )
            
    
# Visualization Car VS Not-Car
# pick a random image from cars
random_pick = np.random.randint(0, len(cars))
random_car = cars[random_pick]
car_image = mpimg.imread(random_car)
car_image_converted = convert_color(car_image) #default is YCrCb
print('randomly picked car: {}'.format(random_car))
# pick a random image from notcars
random_pick = np.random.randint(0, len(notcars))
random_notcar = notcars[random_pick]
notcar_image = mpimg.imread(random_notcar)
notcar_image_converted = convert_color(notcar_image) #default is YCrCb
print('randomly picked not-car: {}'.format(random_notcar))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(car_image)
ax1.set_title('Car', fontsize=20)
ax2.imshow(notcar_image)
ax2.set_title('Not-Car', fontsize=20)
fig.savefig('./output_images/car_not_car.png')


# Visualization HOG features with 3 channels from YCrCb
for i in range(3):

    channel = car_image_converted[:,:,i]  
    features, hog_image = get_hog_features(channel, orient, 
                            pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(channel, cmap='gray')
    car_title = 'Car CH-' + str(i)
    ax1.set_title(car_title, fontsize=20)
    ax2.imshow(hog_image, cmap='gray')
    hog_title = 'HOG_Visualization_Car_CH-' + str(i)
    ax2.set_title(hog_title, fontsize=20)
    filename = './output_images/' + hog_title + '.png'
    fig.savefig(filename)


for i in range(3):
    channel = notcar_image_converted[:,:,i]  
    features, hog_image = get_hog_features(channel, orient, 
                            pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(channel, cmap='gray')
    car_title = 'Not-Car CH-' + str(i)
    ax1.set_title(car_title, fontsize=20)
    ax2.imshow(hog_image, cmap='gray')
    hog_title = 'HOG_Visualization_Not-Car_CH-' + str(i)
    ax2.set_title(hog_title, fontsize=20)
    filename = './output_images/' + hog_title + '.png'
    fig.savefig(filename)