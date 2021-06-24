# Filename: kaggle_covid_models.py
# Description: a playground to get 
# started with transfer learning
# 
# 2021-06-22

# Importing models
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage.transform import resize
from covid_models import DenseNet
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from PIL import Image 
import matplotlib.pyplot as plt
import shutil
import numpy
import cv2
import os

def buildModel():

    # initializing model with access to weights
    dense_init = DenseNet("/data/covid_weights/DenseNet_224_up_crop.h5");
            
    # building base model
    dense_built = dense_init.buildBaseModel(224);
    
    # freezing all layers but the last
    dense_built = dense_init.freeze(dense_built);
    
    # editing last layer to be four class model and creating new model
    dense_kbuilt = Model(inputs = dense_built.input, outputs = Dense(4,
        activation = "sigmoid", name="last")(dense_built.layers[-2].output));
    # returning model
    return dense_kbuilt;

def generateImages(read_dir, write_dir):
    
    def save_as_png(path_to_dcm, path_to_write, size = 512):

        # generating cropped images using image utils
        xray_dicom = pydicom.filereader.dcmread(path_to_dcm);
        # applying volume of interest look up table colors to get opacity in
        # accordance with dicom image format
        xray = apply_voi_lut(xray_dicom.pixel_array, xray_dicom);
        
        # fixing inversion stuff if needed
        if(xray_dicom.PhotometricInterpretation == "MONOCHROME1"):
            xray = numpy.amax(xray) - xray;
        
        # normalizing
        xray = xray - numpy.min(xray);
        xray = xray / numpy.max(xray);
   
        # converting to 8 bit unsigned integer (from gray scale 0 to 1)
        xray = (xray * 255).astype(numpy.uint8);

        # resizing
        xray = resize(xray, (size, size), anti_aliasing = True);
        
        # getting split path for filename
        path_split = path_to_dcm.split("/");

        # generating filename
        filename = "{}-{}-{}".format(path_split[-3], path_split[-2],
                os.path.splitext(path_split[-1])[0]);

        # writing image
        plt.imsave(os.path.join(path_to_write, "{}.png".format(filename)), xray, cmap = "gray", format = "png");

    # checking if path is a directory
    if(not os.path.isdir(read_dir)):
        # giving error message
        print("The path to the directory does not exist.");
        # exiting 
        return;
    if(not os.path.isdir(os.path.join(read_dir, "train")) or not os.path.isdir(os.path.join(read_dir, "test"))):
        # giving error message
        print("The given directory must contain a train and test directory");
        # exiting
        return;
    
    # if write directory exists delete it
    if(os.path.isdir(write_dir)):
        shutil.rmtree(write_dir);

    # creating write directory
    os.makedirs(os.path.join(write_dir, "train"));
    os.makedirs(os.path.join(write_dir, "test"));
    
    print("Converting training dicom files to png...\n");

    # iterating over training data
    for subdir, dirs, files in os.walk(os.path.join(read_dir, "train")):
        for file in files:
            save_as_png(os.path.join(subdir, file), os.path.join(write_dir,
                "train"), 512);
    
    print("Converting training dicom files to png...\n");

    # iterating over testing data 
    for subdir, dirs, files in os.walk(os.path.join(read_dir, "test")):
        for file in files:
            save_as_png(os.path.join(subdir, file), os.path.join(write_dir,
                "test"), 512);

generateImages("/data/kaggle_data/", "/data/_kaggle_data/");
