# Filename: kaggle_create_graphs.py
# Description: This file is a script that
# creates graphs to evaluate models
# 
# 2021-07-08

# importing tools
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense
from vis.visualization import visualize_cam
from vis.visualization import overlay
from vis.utils import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from helper import model_dict, weight_dict
from PIL import Image
import argparse
import numpy
import sys
import os

def create_graphs(args):

    # checking that read dir exists
    if(args.read_dir == None or not os.path.isdir(args.read_dir)):
        # throwing error and exiting
        sys.exit("The directory of the model history does not exist.");
    
    # getting model name
    model_name = args.read_dir.split("/")[-1];

    # getting history path
    hist_path = "{}/{}_history.npy".format(args.read_dir, model_name);

    # checking if history path exists
    if(not os.path.isfile(hist_path)):
        # exiting with error if file doesn't exist
        sys.exit("Something went wrong with reading the model's history");

    # opening file
    hist = numpy.load(hist_path, allow_pickle = True).item();

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
    l_fig.savefig("{}/loss_{}_plot.png".format(args.read_dir, model_name));
    a_fig.savefig("{}/accuracy_{}_plot.png".format(args.read_dir, model_name));


def create_gradCAMs(args):

    # assuming list of model names in directory to be given
    model_names = [
        "dense",
        "inception",
        "inceptionr",
        "resnet",
        "xception"
    ];

    # reading image
    img = Image.open("/data/kaggle_data/class_formatted/test/atypical_appearance/7df6ff1a1957-2b6cbdaf319f-29a7faf526ba.png");

    # numpy
    img = numpy.array(img.convert("RGB"));

    # initializing visualizations vector
    visualizations = [];

    # iterating over model names
    for name in model_names:

        # initializing model with access to weights
        init = model_dict[name]("/data/covid_weights/{}_224_up_crop.h5".format(weight_dict[name]));
                    
        # building base model
        built = init.buildBaseModel(224);
            
        # editing last layer to be four class model and creating new model
        model = Model(inputs = built.input, outputs = Dense(4,
            activation = "softmax", name="last")(built.layers[-2].output));

        # loading weights
        model.load_weights("{}/{}/{}-final.h5".format(args.read_dir, name, name));

        # creating visualization
        n_layers = len(model.layers);

        visualization = visualize_cam(model,
                            layer_idx = n_layers - 1,
                            filter_indices = 0,
                            seed_input = img,
                            penultimate_layer_idx = n_layers - 2);
        
        # saving visualizations
        visualizations.append(visualization);

    # getting average of grad cams
    visualization = visualizations[0] #numpy.mean(visualizations, axis = 0);


    #  plotting grad cam
    plt.rcParams["figure.figsize"] = (18, 6);

    # initializing subplots
    fig, axes = plt.subplots(1, 3);
    
    # plotting
    axes[0].imshow(img[..., 0], cmap = "bone");
    axes[0].set_title("Input");
    axes[1].imshow(visualization);
    axes[1].set_title("Grad-CAM");

    # heatmap over og
    hm = numpy.uint8(cm.jet(visualization)[..., :3] * 255);
    og = numpy.uint8(cm.bone(img[..., 0])[..., :3] * 255);

    axes[2].imshow(overlay(hm, og));
    axes[2].set_title("Overlay");

    # saving figure
    plt.savefig("/data/DeepCovidXR/grad_cam.png");

def create_distribution_graph(args):

    # defining classes
    classes = ["atypical_appearance", "indeterminate_appearance", "no_pneumonia", "typical_appearance"];
    counts = [0, 0, 0, 0];

    # iterating over data
    for folder in os.listdir("/data/kaggle_data/class_formatted"):
        # iterating sub folder
        # for sub_folder in os.listdir("/data/kaggle_data/class_formatted/{}".format(folder)):
        for indx, _class in enumerate(classes):
            print(len(os.listdir("/data/kaggle_data/class_formatted/{}/{}".format(folder, _class))), _class, folder);
            counts[indx] = counts[indx] + len(os.listdir("/data/kaggle_data/class_formatted/{}/{}".format(folder, _class)));
    # creating fig
    fig = plt.figure();
    labels = ["Atypical", "Indeterminate", "No Pneumonia", "Typical"];
    
    plt.bar(labels, counts);
    plt.xlabel("Class");
    plt.ylabel("Class Count");
    plt.title("Data Distribution");
    plt.savefig("data_distribution.png");


if(__name__ == "__main__"):
   
    # initializing parser
    parser = argparse.ArgumentParser();

    # adding arguments
    parser.add_argument("-r", "--read_dir", type = str, help = "The directory where the model history resides.");
    parser.add_argument("-m", "--model", type = str, help = "The name of the model (either dense, xception, resnet, efficient, inception, or inceptionr)");
    parser.add_argument("-f", "--function", type = str, help = "The name of the function that is to be executed.");
    parser.add_argument("-w", "--write_dir", type = str, help = "The directory that graphs are written to.");

    # getting arguments
    args = parser.parse_args();

    # creating function dictionary
    func_dict = {
            "create_graphs": create_graphs,
            "create_gradCAMs": create_gradCAMs,
            "create_distribution_graph": create_distribution_graph
    };

    # checking that function is specified
    if(args.function == None):
        # exiting with error
        sys.exit("A function must be specified to use the script, use the -h or --help flag to learn which arguments to use");

    # executing function
    func_dict[args.function](args);
