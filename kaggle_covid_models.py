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
from utils import imgUtils
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
import tensorflow
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
    
    # editing last layer to be four class model and creating new model
    dense_kbuilt = Model(inputs = dense_built.input, outputs = Dense(4,
        activation = "softmax", name="last")(dense_built.layers[-2].output));


    # returning model
    return dense_kbuilt, dense_init;

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
    
    # keeping track of classes
    classes = ["no_pneumonia", "typical_appearance", "indeterminate_appearance", "atypical_appearance"];

    # keeping track of directories
    dirs = ["train", "valid", "test"];
    
    # creating directories
    for _class in classes:
        for _dir in dirs:
            # writing directory
            os.makedirs(os.path.join(write_dir, _dir, _class));

    # reading in labels
    labels = pandas.read_csv(labels_path);

    # initializing index dictionary where each index corresponds to a class
    class_index_dic = { 0:"no_pneumonia", 1:"typical_appearance",
            2:"indeterminate_appearance", 3:"atypical_appearance" };
    
    # getting list of filenames in data directory
    files = os.listdir(os.path.join(read_dir, "train"));

    # saving number of files
    n_files = len(files);

    # iterating over training data
    for index, file in enumerate(files):

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
        
        # determining where to save
        save_location = None;

        if(index <= n_files / 5):
            save_location = "valid";
        elif(index <= (2 * n_files) / 5):
            save_location = "test";
        else:
            save_location = "train"

        # saving img
        img.save(os.path.join(write_dir, save_location,
            class_index_dic[sample_class], file));

def train_model(read_dir):

    # defining strat
    strategy = tensorflow.distribute.MirroredStrategy();
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync));

    with strategy.scope():
        # making directories
        train_dir = os.path.join(read_dir, "train");
        valid_dir = os.path.join(read_dir, "valid");
        test_dir = os.path.join(read_dir, "test");
        

        # making data generators
        train_datagen = ImageDataGenerator(
            zoom_range = 0.05,
            brightness_range = [0.8, 1.2],
            fill_mode = "constant",
            horizontal_flip = True,
        );

        test_datagen = ImageDataGenerator();
        
        # creating data flows 
        train_gen = train_datagen.flow_from_directory(train_dir, 
                                                      target_size = (224, 224), 
                                                      class_mode = "categorical", 
                                                      color_mode="rgb", 
                                                      batch_size = 16,
                                                      interpolation="lanczos",
                                                      );


        valid_gen = test_datagen.flow_from_directory(valid_dir, 
                                                      target_size = (224, 224), 
                                                      class_mode = "categorical", 
                                                      color_mode="rgb", 
                                                      batch_size = 16,
                                                      interpolation="lanczos",
                                                      );

        test_gen = test_datagen.flow_from_directory(test_dir, 
                                                      target_size = (224, 224), 
                                                      class_mode = "categorical", 
                                                      color_mode="rgb", 
                                                      batch_size = 16,
                                                      interpolation="lanczos",
                                                    );
        # generating weights dictionary

        # getting number of files in training directory
        n_cat_files = int(sum([len(files) for r, d, files in os.walk(valid_dir)]) / 4);

        # initialzing loss weights
        weights_dict = {};

        for key in valid_gen.class_indices.keys():
            
            index = valid_gen.class_indices[key]; # getting index
            # generating weight
            weights_dict[index] = n_cat_files / len(os.listdir(os.path.join(valid_dir, key)));
            print(key, weights_dict[index], len(os.listdir(os.path.join(valid_dir, key))));
    

    if(not os.path.isfile("/data/dense.h5")):
        with strategy.scope():
            # building model
            model, init_model = build_model();

            # freezing model
            model = init_model.freeze(model);

            # compiling
            model.compile(loss = "categorical_crossentropy",
                                 optimizer = SGD(lr = 0.0001),
                                 metrics = ["accuracy"],
                                 );
        # training model
        history = model.fit(
                train_gen, 
                epochs = 10, 
                validation_data = valid_gen,
                class_weight = weights_dict
        );

        # saving model
        model.save("/data/dense.h5");

    with strategy.scope():
        # loading model anew
        base_model = DenseNet121(weights = "imagenet", include_top = False, 
                                    input_shape = (224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(4, activation = "softmax", name = "last")(x)
        model = Model(inputs = base_model.input, outputs = predictions)
        model.load_weights("/data/dense.h5")

        # compiling
        model.compile(loss = "categorical_crossentropy",
                             optimizer = SGD(lr = 0.0001),
                             metrics = ["accuracy"],
                             );
    # training
    history = model.fit(
            train_gen, 
            epochs = 30, 
            validation_data = valid_gen,
            class_weight = weights_dict
    );
    
    # testing model
    accuracy = model.evaluate(test_gen);

    print(accuracy);

# print(list_physical_devices("GPU"));
train_model("/data/covid_xrays");
# resize_organize_images("/data/kaggle_data/train_study_level.csv",
#         "/data/_kaggle_data", "/data/covid_xrays");
