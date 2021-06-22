# Filename: kaggle_covid_models.py
# Description: a playground to get 
# started with transfer learning
# 
# 2021-06-22

# Importing models
from covid_models import DenseNet
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense

# initializing model with access to weights
dense_init = DenseNet("/data/covid_weights/DenseNet_224_up_crop.h5");
        
# building base model
dense_built = dense_init.buildBaseModel(224);

# freezing all layers but the last
dense_built = dense_init.freeze(dense_built);

# editing last layer to be four class model and creating new model
dense_kbuilt = Model(inputs = dense_built.input, outputs = Dense(4,
    activation = "sigmoid", name="last")(dense_built.layers[-2].output));

print(dense_kbuilt.summary())
print(len(dense_kbuilt.layers))
