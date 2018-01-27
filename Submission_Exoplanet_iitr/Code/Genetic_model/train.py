import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#Import libraries for Deep Learning
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam

def load_data():
    # Set defaults.
    nb_classes = 2

    data = pd.read_csv('ExoTrain.csv')
    Y = data['LABEL'].values - 1
    X = data.drop('LABEL', axis=1).values
    Y = Y[:,np.newaxis]
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=531, stratify=Y)	
    
    
    #Scale each observation to zero mean and unit variance.
    x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / 
               np.std(x_train, axis=1).reshape(-1,1))
    x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / 
              np.std(x_test, axis=1).reshape(-1,1))
    
    #Noise in the data can be ignored by adding a gaussian filter
    #This could more elegantly be done as an extra layer.
    x_train = np.stack([x_train, gaussian_filter1d(x_train, 1, axis=1)], axis=2)
    x_test = np.stack([x_test, gaussian_filter1d(x_test, 1, axis=1)], axis=2)

    input_shape = x_train.shape[1:]
    return (nb_classes, input_shape, x_train, x_test, y_train, y_test)


def create_model(network, nb_classes, input_shape):
    """Create and compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    activation = network['activation']
    kernel1= network['kernel1']
    kernel2= network['kernel2']
    kernel3= network['kernel3']
    filter1= network['filter1']
    filter2= network['filter2']
    filter3= network['filter3']
    dropout= network['dropout']
    batch_size = network['batch_size']
    
    
    model = Sequential()
    
    if nb_layers == 3: 
        model.add(Conv1D(filters=filter1, kernel_size=kernel1, activation=activation, input_shape=input_shape))
        model.add(MaxPool1D(strides=4))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=filter2, kernel_size=kernel2, activation=activation))
        model.add(MaxPool1D(strides=4))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=filter3, kernel_size=kernel3, activation=activation))
        model.add(MaxPool1D(strides=4))
        
    else:
        model.add(Conv1D(filters=filter1, kernel_size=kernel1, activation=activation, input_shape=input_shape))
        model.add(MaxPool1D(strides=4))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=filter2, kernel_size=kernel2, activation=activation))
        model.add(MaxPool1D(strides=4))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=filter3, kernel_size=kernel3, activation=activation))
        model.add(MaxPool1D(strides=4))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=filter3, kernel_size=kernel3, activation=activation))
        model.add(MaxPool1D(strides=4))
        

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation=activation))

    # Output layer.
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model


# feeding training data to the model in batches
def batch_generator(x_train, y_train, batch_size):
    """
    Gives equal number of positive and negative samples, and rotates them randomly in time
    """
    half_batch = 16
    
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')
    
    planet = np.where(y_train[:,0] == 1.)[0]
    non_planet = np.where(y_train[:,0] == 0.)[0]
    
    while True:
        np.random.shuffle(planet)
        np.random.shuffle(non_planet)
    
        x_batch[:half_batch] = x_train[planet[:half_batch]]
        y_batch[:half_batch] = y_train[planet[:half_batch]]
        x_batch[half_batch:] = x_train[non_planet[half_batch:batch_size]]
        y_batch[half_batch:] = y_train[non_planet[half_batch:batch_size]]
    
        for i in range(batch_size):
            roll_state = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], roll_state, axis = 0)
     
        yield x_batch, y_batch
        
def training_and_evaluation(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network

    """
            
    nb_classes, input_shape, x_train, x_test, y_train, y_test = load_data()   
    batch_size = network['batch_size']
  
    model = create_model(network, nb_classes, input_shape)

    model.fit_generator(batch_generator(x_train, y_train, batch_size), 
                        validation_data=(x_test, y_test),
                        verbose=2, epochs=35,
                        steps_per_epoch=x_train.shape[1]//32)
    
    model.compile(optimizer=Adam(0.0001), loss = 'binary_crossentropy', metrics=['accuracy'])
    model.fit_generator(batch_generator(x_train, y_train, batch_size), 
                        validation_data=(x_test, y_test),
                        verbose=2, epochs=10,
                        steps_per_epoch=x_train.shape[1]//32)

    model.compile(optimizer=Adam(0.00001), loss = 'binary_crossentropy', metrics=['accuracy'])
    model.fit_generator(batch_generator(x_train, y_train, batch_size), 
                        validation_data=(x_test, y_test),
                        verbose=2, epochs=10,
                        steps_per_epoch=x_train.shape[1]//32)
    # f1_score with average macro because this is a highly imbalance class problem
    
    y_pred = model.predict(x_test)[:,0]
    y_true = (y_test[:, 0] + 0.5).astype("int")
    f1_score_= []
    skill_score_= []

    for i in (np.arange(0.5,1.0,.005)):
        y_pred_int= np.copy(y_pred)
        y_pred_int[(y_pred_int>i)]=1
        y_pred_int[(y_pred_int<=i)]=0

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_int).ravel()

        skill_score_.append(((tp * tn) - (fp * fn)) / ((tp + fn) * (fp + tn)))
        f1_score_.append(f1_score(y_true, y_pred_int, average='macro'))


    score = np.max(f1_score_)
    
    print("*"*80)
    print("No. of Layers : ", network['nb_layers'])
    print("Activation type : ", network['activation'])
    print("1st Kernal size : ", network['kernel1'])
    print("2nd Kernal size : ", network['kernel2'])
    print("3rd &/or 4th Kernal size : ", network['kernel3'])
    print("No. of filters in 1st layer : ", network['filter1'])
    print("No. of filters in 2nd layer : ", network['filter2'])
    print("No. of filters in 3th &/or 4th layer : ", network['filter3'])
    print("Dropout fraction : ", network['dropout'])
    print("Batch size : ", batch_size)
    print("*"*80)
    print("F1 score : ", score)
    print("Threshold for F1 score : ", (np.min(np.where(f1_score_ == np.max(f1_score_)))+1)*0.005+0.5)
    print("*"*40)
    print("True skill score : ", np.max(skill_score_))
    print("Threshold for True skill score : ", (np.min(np.where(skill_score_ == np.max(skill_score_)))+1)*0.005+0.5)
    print("*"*80)
    print("*"*80)
    
    return score 
