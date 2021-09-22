# Filename: kaggle_ensemble.py
# Description: This file implements 
# the ensemble from previoulsy trained covid models

# importing tools
from covid_models import DenseNet, XceptionNet, ResNet, EfficientNet, InceptionNet, InceptionResNet
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import numpy
import os

def create_ensemble(ensemble_dir):
    
    # building model dictionary
    model_dict = {
            "dense": DenseNet,
            "xception": XceptionNet,
            "resnet": ResNet,
            "efficient": EfficientNet,
            "inception": InceptionNet,
            "inceptionr": InceptionResNet
    };

    # building weight dictionary
    weight_dict = {
            "dense": ("DenseNet", 0.6354),
            "xception": ("Xception", 0.6464),
            "resnet": ("ResNet50", 0.6314),
            "efficient": "EfficientNet",
            "hyper": None,
            "inception": ("Inception", 0.6117),
            "inceptionr": ("InceptionResNet", 0.6346)
    };

    # initializing ensemble
    ensemble = [];

    # reading over directories in ensemble dir
    for model_name in os.listdir(ensemble_dir):

        # initializing model with access to weights
        init = model_dict[model_name]("/data/covid_weights/{}_224_up_uncrop.h5".format(weight_dict[model_name][0]));

        # building base model
        built = init.buildBaseModel(224);

        # editing last layer to be four class model and creating new model
        kbuilt = Model(inputs = built.input, outputs = Dense(4,
            activation = "softmax", name="last")(built.layers[-2].output));

        # loading real weights
        kbuilt.load_weights("/data/trained_models/{}/{}-final.h5".format(model_name, model_name));

        # appending model to models
        ensemble.append((kbuilt, weight_dict[model_name][1]));

    # retunring ensemble
    return ensemble;

def ensemble_evaluation():

    # getting ensemble 
    ensemble = create_ensemble("/data/wtrained_models");

    # classes
    classes = ["atypical_appearance", "indeterminate_appearance", "no_pneumonia", "typical_appearance"]

    # initiaizing evaluating metrics
    correct = 0;
    incorrect = 0;

    # initializing confusion matrix variables
    gt = [];
    p = [];

    # iterating over testing data
    for root, dirs, files in os.walk("/data/kaggle_data/class_formatted/test/"):

        # iterating over files
        for file in files:
            # creating file path
            file_path = os.path.join(root, file);

            # opening image
            im = Image.open(file_path).convert("RGB");

            # converting to numpy
            im = numpy.asarray(im);
            im = numpy.expand_dims(im, axis = 0)

            # initializing votes
            votes = numpy.zeros(shape = (4));
            
            # iterating over ensemble
            for index, model in enumerate(ensemble):
                # predicting using model
                pred = model[0].predict(im)[0];
                vote = numpy.argmax(pred);
                strength = numpy.max(pred);

                # voting 
                votes[vote] = votes[vote] + (model[1]);

            # appending confusion matrix values
            gt.append(root.split("/")[-1]);
            p.append(classes[numpy.argmax(votes)]);

            # predicting 
            if(classes[numpy.argmax(votes)] == root.split("/")[-1]):
                # indicating correct
                correct = correct + 1;
            else:
                # indicating incorrect
                incorrect = incorrect + 1;

    print("acc", correct / (correct + incorrect));
    conf_mat = multilabel_confusion_matrix(gt, p);
    # intializing accuracy, sensitivity, and specificity arrays
    acc = [];
    sens = [];
    spec = [];

    # iterating over confusion matrix to get values
    for mat in conf_mat:

        # getting accurcacy, sensitivity, and specificity
        tp = mat[1][1];
        fn = mat[1][0];
        fp = mat[0][1];
        tn = mat[0][0];

        # collecting data
        acc.append((tp + tn) / (tp + fn + fp + tn));
        sens.append(tp / (tp + fn));
        spec.append(tn / (tn + fp));

    # priniting avg accuracy
    print("avg acc", sum(acc) / len(acc));
    print("sens", sens, sum(sens) / len(sens));
    print("spec", spec, sum(spec) / len(spec));

    # creating labels
    labels = ["Atypical", "Indeterminate", "No Pneumonia", "Typical"];

    # getting label locations
    x = numpy.arange(len(labels));
    width = 0.4

    # plotting
    fig, ax = plt.subplots();
    ac = ax.bar(x - width / 3, acc, width, label = "Accuracy");
    se = ax.bar(x, sens, width, label = "Sensitivity");
    sp = ax.bar(x + width / 3, spec, width, label = "Specificity");

    # adding labels and titles
    ax.set_ylabel("Scores");
    ax.set_title("Accuracy, Sensitivity, and Specificity");
    ax.set_xticks(x);
    ax.set_xticklabels(labels);
    ax.legend();
    
    # ax.bar_label(ac, padding = 3);
    # ax.bar_label(se, padding = 3);
    # ax.bar_label(sp, padding = 3);

    fig.tight_layout();

    # saving figure
    plt.savefig("confusion_test.png");

ensemble_evaluation();

