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
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.test import gpu_device_name
from tensorflow.config import list_physical_devices
import pandas
from PIL import Image 
import matplotlib.pyplot as plt
import shutil
import numpy
import cv2
import os

def build_model():

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

def generate_images(read_dir, write_dir):
    # Function: generate_images, a function that takes the kaggle dataset and
    # converts the dicoms to pngs. 

    # Warning: This function takes several hours to run.

    # Parameter(s):

    #     read_dir - the directory that should be read from
    #     write_dir - the directory that should be written to

    # Return Value(s):

    #     None

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
    
    # initialzing training counter
    train_count = 0;

    # iterating over training data
    for subdir, dirs, files in os.walk(os.path.join(read_dir, "train")):
        for file in files:
            save_as_png(os.path.join(subdir, file), os.path.join(write_dir,
                "train"), 512);
            train_count = train_count + 1;
            print("Saved {} training images".format(train_count));

    
    print("Converting training dicom files to png...\n");

    # initializing testing counter
    test_count = 0;

    # iterating over testing data 
    for subdir, dirs, files in os.walk(os.path.join(read_dir, "test")):
        for file in files:
            save_as_png(os.path.join(subdir, file), os.path.join(write_dir,
                "test"), 512);
            test_count = test_count + 1;
            print("Saved {} testing images".format(test_count));

def resize_organize_images(labels_path, read_dir, write_dir):
    
    # checking if write directory exists and deleting if it does
    if(os.path.isdir(write_dir)):
        # deleting
        shutil.rmtree(write_dir);

    # creating directories
    os.makedirs(os.path.join(write_dir, "train", "no_pneumonia"));
    os.makedirs(os.path.join(write_dir, "train", "typical_appearance"));
    os.makedirs(os.path.join(write_dir, "train", "indeterminate_appearance"));
    os.makedirs(os.path.join(write_dir, "train", "atypical_appearance"));

    # reading in labels
    labels = pandas.read_csv(labels_path);

    # initializing index dictionary where each index corresponds to a class
    class_index_dic = { 0:"no_pneumonia", 1:"typical_appearance",
            2:"indeterminate_appearance", 3:"atypical_appearance" };

    # iterating over training data
    for file in os.listdir(os.path.join(read_dir, "train")):

        # getting full path
        full_path = os.path.join(read_dir, "train", file);

        # opening image
        img = Image.open(full_path);

        # resizing
        img = img.resize((224, 224));

        # getting study id
        study_id = file.split("-")[0];

        # getting csv id
        csv_id = "{}_study".format(study_id);
       
        # getting sample class
        sample_class = labels.iloc[labels.index[labels["id"] ==
            csv_id].tolist()[0]].to_numpy()[1:].argmax();

        # saving img
        img.save(os.path.join(write_dir, "train",
            class_index_dic[sample_class], file));

def train_model(train_dir):

    # building model
    model = build_model();

    # getting training data
    train_set = image_dataset_from_directory(train_dir, seed = 5, image_size
            = (224, 224), batch_size = 32);

    # compiling model
    model.compile(loss = SparseCategoricalCrossentropy(from_logits=False), optimizer = "rmsprop", metrics
            = ["accuracy"]);

    # training model
    history = model.fit(train_set, epochs = 8);


print(list_physical_devices("GPU"));
#train_model("/data/covid_xrays/train");
# resize_organize_images("/data/kaggle_data/train_study_level.csv",
#         "/data/_kaggle_data", "/data/covid_xrays");
