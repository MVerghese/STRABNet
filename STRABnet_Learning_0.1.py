#Creates a new model
MODEL_CREATE = False
#Data csv
MODEL_DATA = 'STRABnet_Artifical_Classifier.csv'
#Saves a new model
MODEL_SAVE = False
#Model save name
MODEL_SAVE_NAME = 'STRABnet_model.h5'
#Loads an existing model (if set to false, will use created model)
MODEL_LOAD = True
#Model to load
MODEL_LOAD_NAME = 'STRABnet_model_V1.h5'
#Run predictions
PREDICT = True
#Prediction data
PRED_DATA_NAME = "STRABnet_Patient_Data.csv"


# Import necessary modules
print("Loading")
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras import optimizers
import pandas as pd

print("Complete")

if(MODEL_CREATE):
    data = pd.read_csv(MODEL_DATA)

    predictors = data.drop(['Class'], axis=1).as_matrix()

    n_cols = predictors.shape[1]

    # Convert the target to categorical: target
    target = to_categorical(data.Class)

    # Set up the model
    model = Sequential()

    # Add the first layer
    model.add(Dense(100, activation='relu', input_shape=(n_cols,)))

    #Add the hidden layers
    model.add(Dense(100, activation='relu'))

    model.add(Dense(100, activation='relu'))

    # Add the output layer
    model.add(Dense(9, activation='softmax'))

    # Compile the model
    opt = optimizers.SGD(lr=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(predictors, target, epochs=50)

if(MODEL_SAVE):
    model.save(MODEL_SAVE_NAME)

if(MODEL_LOAD):
    model = load_model(MODEL_LOAD_NAME)

if(PREDICT):
    raw_pred_data = pd.read_csv(PRED_DATA_NAME)
    pred_data = raw_pred_data.as_matrix()
    
    
    

    # Calculate predictions: predictions
    predictions = model.predict(pred_data)

    # Calculate predicted probability of survival: predicted_prob_true
    predicted_prob_true = predictions[:,1]

    # print predicted_prob_true
    print(predictions)
    counter = 1
    for case in predictions:
        diagnosis = 0
        for i in range(0,9):
            if case[i] > case[diagnosis]:
                diagnosis = i
        if(case[diagnosis] > .25):
            print(counter, diagnosis)
        else:
            print(counter, "No Significant answer")
        counter = counter + 1
            
