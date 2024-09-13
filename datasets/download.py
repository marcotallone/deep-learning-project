# Script to download the necessary datasets for the project from Kaggle

# Imports -----------------------------------------------------------------------
import os
import shutil
import numpy as np 
from tqdm import tqdm
import cv2
import os
import imutils
from kaggle.api.kaggle_api_extended import KaggleApi


# Initialize Kaggle API --------------------------------------------------------
# To use this: 
#   1. Go to your Kaggle account settings and create a new API token. This will download a kaggle.json file.
#   2. Place the kaggle.json file in the ~/.kaggle/ directory (create the directory if it doesn't exist).
api = KaggleApi()
api.authenticate()


# Create the `classification/` and `segmentation/` directories -----------------
# if they don't exist
if not os.path.exists('./classification'):
    os.makedirs('./classification')
if not os.path.exists('./segmentation'):
    os.makedirs('./segmentation')


# Dataset 1: Brain MRI Images for Multicalss Tumor Classification --------------
# URL: https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset
dataset1 = 'masoudnickparvar/brain-tumor-mri-dataset'
print("Downloading dataset 1...")
api.dataset_download_files(dataset1, path='./classification/', unzip=True)
print("Dataset 1 downloaded successfully.")

# Cleaning dataset 1 images (cropping white borders and resizing into 256x256x3)
def crop_img(img):
	"""
	Finds the extreme points on the image and crops the rectangular out of them
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)

	# threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
	thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	# find contours in thresholded image, then grab the largest one
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# find the extreme points
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])
	ADD_PIXELS = 0
	new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
	
	return new_img
	
training = "classification/Training"
testing = "classification/Testing"
training_dir = os.listdir(training)
testing_dir = os.listdir(testing)
IMG_SIZE = 256
CLEANING_DIR='classification/cleaned'
print("Cleaning dataset 1 images...")

if not os.path.exists(CLEANING_DIR):
    os.makedirs(CLEANING_DIR)

for dir in training_dir:
    save_path = 'classification/cleaned/Training/'+ dir
    path = os.path.join(training,dir)
    image_dir = os.listdir(path)
    for img in image_dir:
        image = cv2.imread(os.path.join(path,img))
        new_img = crop_img(image)
        new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path+'/'+img, new_img)

for dir in testing_dir:
    save_path = 'classification/cleaned/Testing/'+ dir
    path = os.path.join(testing,dir)
    image_dir = os.listdir(path)
    for img in image_dir:
        image = cv2.imread(os.path.join(path,img))
        new_img = crop_img(image)
        new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path+'/'+img, new_img)

# Remove original images and move cleaned images to the root directory
print("Substituting original images with cleaned ones...")
os.system("rm -rf classification/Training")
os.system("rm -rf classification/Testing")
os.system("mv classification/cleaned/* classification/")
os.system("rm -rf classification/cleaned")


# Dataset 2: BraTS2020 dataset for Brain Tumor Segmentation --------------------
# URL: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data
dataset2 = 'awsaf49/brats2020-training-data'
print("Downloading dataset 2...")
api.dataset_download_files(dataset2, path='./segmentation/', unzip=True)
print("Dataset 2 downloaded successfully.")

# Create the `segmentation/data/` directory if it doesn't exist
if not os.path.exists('segmentation/data'):
    os.makedirs('segmentation/data')

print("Reorganizing segmentation directory...")

# Define source and destination directories
source_dir = 'segmentation/BraTS2020_training_data/content/data/'
dest_dir1 = 'segmentation/data/'
dest_dir2 = 'segmentation/'

# Ensure destination directories exist
os.makedirs(dest_dir1, exist_ok=True)
os.makedirs(dest_dir2, exist_ok=True)

# Move .h5 files to segmentation/data/
for file_name in os.listdir(source_dir):
    if file_name.endswith('.h5'):
        shutil.move(os.path.join(source_dir, file_name), dest_dir1)

# Move all other files to segmentation/
for file_name in os.listdir(source_dir):
    if not file_name.endswith('.h5'):
        shutil.move(os.path.join(source_dir, file_name), dest_dir2)

# Remove the empty source directory
os.rmdir('segmentation/BraTS2020_training_data')