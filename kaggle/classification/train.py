# Filename: kaggle_covid_models.py
# Description: a playground to get 
# started with transfer learning
# 
# 2021-06-22

# Importing models
import argparse
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from utils import imgUtils
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.metrics import AUC, Precision, Recall
from helper import model_dict, weight_dict, params_dict
import tensorflow
import shutil
import numpy
import json
import cv2
import sys
import os

def build_model(model_name):

    # initializing model with access to weights
    init = model_dict[model_name]("/data/covid_weights/{}_224_up_uncrop.h5".format(weight_dict[model_name]));
            
    # building base model
    built = init.buildBaseModel(224);
    
    # editing last layer to be four class model and creating new model
    kbuilt = Model(inputs = built.input, outputs = Dense(4,
        activation = "softmax", name="last")(built.layers[-2].output));


    # returning model
    return kbuilt, init;


def train_model(args):

    # making directories
    train_dir = os.path.join(args.data, "train");
    valid_dir = os.path.join(args.data, "valid");
    test_dir = os.path.join(args.data, "test");
        

    # making data generators
    train_datagen = ImageDataGenerator(
        zoom_range = 0.05,
        brightness_range = [0.8, 1.2],
        fill_mode = "constant",
        horizontal_flip = True,
    );

    test_datagen = ImageDataGenerator();
        
    # creating data flows 
    train_gen = train_datagen.flow_from_directory(
                                                train_dir, 
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

    # getting number of files in training directory
    n_cat_files = int(sum([len(files) for r, d, files in os.walk(valid_dir)]) / 4);

    # initialzing loss weights
    weights_dict = {};

    # collecting percentages and keys
    prcs = [];
    keys = [];

    for key in valid_gen.class_indices.keys():
            
        keys.append(valid_gen.class_indices[key]); # getting index
        # generating weight
        prcs.append(len(os.listdir(os.path.join(valid_dir, key))) / n_cat_files);
    
    # normalizing
    prcs = [(prc - min(prcs)) / max(prcs) for prc in prcs];

    # setting weights dictionary
    for index, prc in zip(keys, prcs):
        # setting weight
        weights_dict[index] = 222**(-1 * prc + (7.5 / 8));
        print(prc, weights_dict[index], index); 
   
    # deleting write if exists
    if(os.path.isdir(args.write)):
        shutil.rmtree(args.write);

    # making write
    os.makedirs(args.write);

    if(not os.path.isfile(os.path.join(args.write, "{}-pre.h5".format(args.model)))):
        
        # building model
        model, init_model = build_model(args.model);

        # freezing model
        model = init_model.freeze(model);

        # compiling
        model.compile(loss = "categorical_crossentropy",
                optimizer = SGD(lr = params_dict[args.model]),
                metrics = ["accuracy", AUC(name = "auc"), Precision(name = "prec"), Recall(name = "rec")],
        );

        # training model
        history = model.fit(
                train_gen, 
                epochs = 20, 
                validation_data = valid_gen,
                class_weight = weights_dict
        );

        # saving model
        model.save(os.path.join(args.write, "{}-pre.h5".format(args.model)));

        
    # initializing model with access to weights
    init = model_dict[args.model]("/data/covid_weights/{}_224_up_uncrop.h5".format(weight_dict[args.model]));
                
    # building base model
    built = init.buildBaseModel(224);
        
    # editing last layer to be four class model and creating new model
    model = Model(inputs = built.input, outputs = Dense(4,
    activation = "softmax", name="last")(built.layers[-2].output));
    model.load_weights(os.path.join(args.write, "{}-pre.h5".format(args.model)));

    # compiling
    model.compile(loss = "categorical_crossentropy",
                    optimizer = SGD(lr = params_dict[args.model]),
                    metrics = ["accuracy", AUC(name = "auc"), Precision(name = "prec"), Recall(name = "rec")],
                );
    # training
    history = model.fit(
            train_gen, 
            epochs = 50, 
            validation_data = valid_gen,
            class_weight = weights_dict
    );
    
    # testing model
    accuracy = model.evaluate(test_gen);

    model.save(os.path.join(args.write, "{}-final.h5".format(args.model)));
    model.save(args.write);

    # saving history
    numpy.save(os.path.join(args.write, "{}_history.npy".format(args.model)), history.history);



if(__name__ == "__main__"):
    # This is the main function of the training script
    
    # initializing parser
    parser = argparse.ArgumentParser();

    # adding arguments
    parser.add_argument("-d", "--data", type = str, help = "Where the data lives (in designated file structure)");
    parser.add_argument("-f", "--function", type = str, help = "Either train_model (for one indidual model) or train_ensemble (for all models)");
    parser.add_argument("-m", "--model", type = str, help = "Model you wish to individually train (dense, efficient, hyper, inception, inceptionr, resnet, or xception)");
    parser.add_argument("-w", "--write", type = str, help = "Directory that weights are written to");
    parser.add_argument("-lw", "--load_weights", type = str, help = "Where weights (for transfer learning) are loaded from");

    # parsing arguments
    args = parser.parse_args();

    # initializing function dictionary
    functions = { "train_model": train_model, "train_ensemble": None };

    # checking if function specified
    if(args.function == None):
        sys.exit("A function must be provided. Use the -h or --help flag to learn to use this script.");
    
    # executing script
    functions[args.function](args);

