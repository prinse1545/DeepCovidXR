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
import numpy
import cv2

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

def generateImages():

    # generating cropped images using image utils
    xray_dicom = pydicom.filereader.dcmread("/data/kaggle_data/train/6263c19334f8/1acf610fbe04/dfc7b258312f.dcm");
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
    xray = resize(xray, (224, 224), anti_aliasing = True);

    # writing image
    plt.imsave("test.png", xray, cmap = "gray", format = "png");


generateImages();
