# Filename: kaggle_create_graphs.py
# Description: This file is a script that
# creates graphs to evaluate models
# 
# 2021-07-08

# importing tools
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
    a_fig.savefig("/data/accuracy_inception_plot.png");


makeGraphs("/data/inception_weights/inception_history.npy");
