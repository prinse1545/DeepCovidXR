# Filename: kaggle_create_graphs.py
# Description: This file is a script that
# creates graphs to evaluate models
# 
# 2021-07-08

# importing tools
from covid_models import DenseNet, XceptionNet, ResNet, EfficientNet, InceptionNet, InceptionResNet
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense
from vis.visualization import visualize_cam
from vis.utils import utils
import matplotlib.pyplot as plt
import numpy
import os

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
    l_fig.savefig("/data/loss_inception_plot.png")
    a_fig.savefig("/data/accuracy_xception_plot.png");


def create_gradCAMs(models_dir):

    # assuming list of model names in directory to be given
    model_names = [
        "dense",
        "inception",
        "inceptionr",
        "resnet",
        "xception"
    ];

    # building model dictionary
    model_dict = { 
            "dense": ("DenseNet", DenseNet), 
            "xception": ("Xception", XceptionNet),
            "resnet": ("ResNet50", ResNet),
            "efficient": ("EfficientNet", EfficientNet),
            "inception": ("Inception", InceptionNet),
            "inceptionr": ("InceptionResNet", InceptionResNet)
    };
        

    # iterating over model names
    for name in model_names:

        # initializing model with access to weights
        init = model_dict[name][1]("/data/covid_weights/{}_224_up_crop.h5".format(model_dict[name][0]));
                    
        # building base model
        built = init.buildDropModel(224, 0.15);
            
        # editing last layer to be four class model and creating new model
        model = Model(inputs = built.input, outputs = Dense(4,
            activation = "softmax", name="last")(built.layers[-2].output));

        # loading weights
        model.load_weights("{}/{}/{}-final.h5".format(models_dir, name, name));

        # creating visualization
        n_layers = len(model.layers);

        visualize_cam(model,
                    layer_idx = n_layers - 1,
                    filter_indices = 0,
                    seed_input = None,
                    penultimate_layer_idx = n_layers - 2,
                    backprop_modifier = None);

def create_distribution_graph():

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

makeGraphs("/data/wtrained_models/xception/xception_history.npy");
# create_gradCAMs("/data/trained_models/");
# create_distribution_graph();
