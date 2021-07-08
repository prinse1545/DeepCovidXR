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
import matplotlib.pyplot as plt
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
    init = model_dict[model_name]("/data/covid_weights/{}_224_up_crop.h5".format(weight_dict[model_name]));
            
    # building base model
    built = init.buildBaseModel(224);
    
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

    print(accuracy);

    model.save("/data/final.h5");

    # saving history
    numpy.save("model_history.npy", history.history);

def makeGraphs(hist_path):

    # seeing if path to file with history exists
    if(not os.path.isfile(hist_path)):
        # printing error and returning 
        print("The file for printing history does not exist!");

        return;

    # opening file
    hist = numpy.load(hist_path, allow_pickle = True).item();

    print(hist.keys());
    # making x axis
    x_axis = list(range(1, len(hist["accuracy"]) + 1));
   
    # accuracy and loss plots
    a_fig, a_ax = plt.subplots();
    l_fig, l_ax = plt.subplots();

    a_ax.plot(x_axis, hist["accuracy"], "ro", label = "accuracy");
    l_ax.plot(x_axis, hist["loss"], "ro", label = "loss");
    a_ax.plot(x_axis, hist["val_accuracy"], "bo", label = "validation accuracy");
    l_ax.plot(x_axis, hist["val_loss"], "bo", label = "validation loss");

    # setting labels
    a_ax.set_xlabel("epochs");
    l_ax.set_xlabel("epochs");

    a_ax.legend();
    l_ax.legend();
    # saving figure
    l_fig.savefig("/data/loss_plot.png")
    a_fig.savefig("/data/accuracy_plot.png");

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
    parser.add_argiment("w", "--write", type = str, help = "Directory that weights are written to");
    parser.add_argument("lw", "--load_weights", type = str, help = "Where weights (for transfer learning) are loaded from");

    # parsing arguments
    args = parser.parse_args();

    # initializing function dictionary
    functions = { "train_model": train_model, "train_ensemble": None };
    
    # executing script
    functions[args.function](args);

