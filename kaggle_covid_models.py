# Filename: kaggle_covid_models.py
# Description: a playground to get 
# started with transfer learning
# 
# 2021-06-22

# Importing models
import pydicom
from covid_models import DenseNet
from tensorflow.keras.models import Sequential, Model
from skimage.transform import resize
from tensorflow.keras.layers import Dense
from PIL import Image
import numpy

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
    xray = pydicom.filereader.dcmread("/data/kaggle_data/train/005057b3f880/e34afce999c5/3019399c31f4.dcm").pixel_array;

    # resizing dicom image
    xray = resize(xray, (224, 224), anti_aliasing = True);
    
    # converting to PIL
    pil_xray = Image.fromarray(xray.astype(numpy.uint8));
    pil_xray.save("test.png");
    print(xray)
    print(type(xray))

    

generateImages();
