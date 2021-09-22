# Filename: kaggle_covid_models.py
# Description: a playground to get 
# started with transfer learning
# 
# 2021-06-22

# Importing models
import argparse
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from covid_models import DenseNet, XceptionNet, ResNet, EfficientNet, InceptionNet, hyperModel, InceptionResNet
from utils import imgUtils
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.metrics import AUC, Precision, Recall
import tensorflow
import shutil
import numpy
import json
import cv2
import os

def build_model(model_name):

    # building model dictionary
    model_dict = { 
            "dense": DenseNet, 
            "xception": XceptionNet,
            "resnet": ResNet,
            "efficient": EfficientNet,
            "hyper": hyperModel,
            "inception": InceptionNet,
            "inceptionr": InceptionResNet
    };
    # building weight dictionary
    weight_dict = {
            "dense": "DenseNet",
            "xception": "Xception",
            "resnet": "ResNet50",
            "efficient": "EfficientNet",
            "hyper": None,
            "inception": "Inception",
            "inceptionr": "InceptionResNet"
    };

    # initializing model with access to weights
    init = model_dict[model_name]("/data/covid_weights/{}_224_up_uncrop.h5".format(weight_dict[model_name]));
            
    # building base model
    built = init.buildDropModel(224, 0.15);
    
    # editing last layer to be four class model and creating new model
    kbuilt = Model(inputs = built.input, outputs = Dense(4,
        activation = "softmax", name="last")(built.layers[-2].output));


    # returning model
    return kbuilt, init;


def train_model(args):

    # defining strat
    strategy = tensorflow.distribute.MirroredStrategy();
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync));

    with strategy.scope():
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

        # getting number of files in training directory
        n_cat_files = int(sum([len(files) for r, d, files in os.walk(valid_dir)]) / 4);

        # initialzing loss weights
        weights_dict = {};

        for key in valid_gen.class_indices.keys():
            
            index = valid_gen.class_indices[key]; # getting index
            # generating weight
            prc = len(os.listdir(os.path.join(valid_dir, key))) / n_cat_files;
            weights_dict[index] = 222**(-1 * prc + (7.5 / 8)) + 0.5;
            print(prc, weights_dict[index], key); 
   
    # deleting write if exists
    if(os.path.isdir(args.write)):
        shutil.rmtree(args.write);

    # making write
    os.makedirs(args.write);

    if(not os.path.isfile(os.path.join(args.write, "{}-pre.h5".format(args.model)))):
        with strategy.scope():
            # building model
            model, init_model = build_model(args.model);

            # freezing model
            model = init_model.freeze(model);

            # compiling
            model.compile(loss = "categorical_crossentropy",
                                 optimizer = SGD(lr = 0.001),
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

    with strategy.scope():
        # building model dictionary
        model_dict = { 
                "dense": DenseNet, 
                "xception": XceptionNet,
                "resnet": ResNet,
                "efficient": EfficientNet,
                "hyper": hyperModel,
                "inception": InceptionNet,
                "inceptionr": InceptionResNet
        };

        # building weight dictionary
        weight_dict = {
                "dense": "DenseNet",
                "xception": "Xception",
                "resnet": "ResNet50",
                "efficient": "EfficientNet",
                "hyper": None,
                "inception": "Inception",
                "inceptionr": "InceptionResNet"
        };
        
        # initializing model with access to weights
        init = model_dict[args.model]("/data/covid_weights/{}_224_up_uncrop.h5".format(weight_dict[args.model]));
                
        # building base model
        built = init.buildDropModel(224, 0.3);
        
        # editing last layer to be four class model and creating new model
        model = Model(inputs = built.input, outputs = Dense(4,
        activation = "softmax", name="last")(built.layers[-2].output));
        # base_model = DenseNet121(weights = "imagenet", include_top = False, 
        #                             input_shape = (224, 224, 3))
        # x = base_model.output
        # x = GlobalAveragePooling2D()(x)
        # predictions = Dense(4, activation = "softmax", name = "last")(x)
        # model = Model(inputs = base_model.input, outputs = predictions)
        model.load_weights(os.path.join(args.write, "{}-pre.h5".format(args.model)));

        # compiling
        model.compile(loss = "categorical_crossentropy",
                             optimizer = SGD(lr = 0.001),
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


# print(list_physical_devices("GPU"));
# train_model("/data/covid_xrays");
# makeGraphs("/data/model_history.npy");
# resize_organize_images("/data/kaggle_data/train_study_level.csv",
#         "/data/_kaggle_data", "/data/covid_xrays");

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
    
    # executing script
    functions[args.function](args);

