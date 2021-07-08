# Filename: kaggle_data_preprocess.py
# Description: A script for processing
# the kaggle data into a model that can 
# be used by the ensemble.
# 
# 2021-07-08

# importing tools
from skimage.transform import resize
from PIL import Image 
import argparse
import pydicom
import pandas
import numpy
import shutil
import os


def to_png(read_dir, write_dir):
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



def create_filestruct(labels_path, read_dir, write_dir):
    
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


if(__name__ == "__main__"):
    # This is the main function of 
    # the preprocessing script.

    # initializing arparser
    parser = argparse.ArgumentParser();

    # adding arguments
    parser.add_argument("-r", "--read_dir", type = str, help = "Where the script reads from");
    parser.add_argument("-w", "--write_dir", type = str, help = "Where the script writes to");
    parser.add_argument("-f", "--function", type = str, help = "What the script does. Either fetch_dataset, to_png, or create_filestruct");
    parser.add_argument("-l", "--labels", type = str, help = "Where the labels provided by kaggle live (should be a csv");

    # getting args
    args = parser.parse_args();
