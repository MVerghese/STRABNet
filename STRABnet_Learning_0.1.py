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
MODEL_LOAD_NAME = 'STRABnet_model100.h5'
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
from sklearn.preprocessing import MinMaxScaler

print("Complete")

if(MODEL_CREATE):

    #FLoad and format data
    data = pd.read_csv(MODEL_DATA)
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaler.transform(data)


    predictors = data.drop(['Class'], axis=1).as_matrix()
    n_cols = predictors.shape[1]
    target = to_categorical(data.Class)

    #Create model as two layers of 32 hidden nodes and then the output layer
    model = Sequential()
    model.add(Dense(32, activation='sigmoid', input_shape=(n_cols,)))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(9, activation='softmax'))

    # Compile the model
    opt = optimizers.SGD(lr=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(predictors, target, epochs=1000)

if(MODEL_SAVE):
    model.save(MODEL_SAVE_NAME)

if(MODEL_LOAD):
    model = load_model(MODEL_LOAD_NAME)

if(PREDICT):
    raw_pred_data = pd.read_csv(PRED_DATA_NAME)
    pred_data = raw_pred_data.as_matrix()
    
    
    

    # Calculate predictions: predictions
    predictions = model.predict(pred_data)
    predicted_prob_true = predictions[:,1]

    # print predicted_prob_true
    print(predictions)
    counter = 1
    for case in predictions:
        diagnosis = 0
        for i in range(0,9):
            if case[i] > case[diagnosis]:
                diagnosis = i
        if(case[diagnosis] > 0):
            print(counter, diagnosis)
        else:
            print(counter, "No Significant answer")
        counter = counter + 1
            
