"""
   train the RamanNet model
"""

from data_processing import segment_spectrum_batch
from RamanNet_model import RamanNet
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau 
from keras.metrics import CategoricalAccuracy


def train_model(X_train, Y_train_onehot, X_val, Y_val_onehot, w_len, dw, epochs, model_path, plot=True):
    
    Y_train = np.argmax(Y_train_onehot,axis=1)
    Y_val = np.argmax(Y_val_onehot,axis=1)

    X_train = segment_spectrum_batch(X_train, w_len, dw)
    X_val = segment_spectrum_batch(X_val, w_len, dw)

    mdl = RamanNet(X_train[0].shape[1],len(X_train), np.max(Y_train)+1)

    losses = {
	    "embedding": tfa.losses.TripletSemiHardLoss(),
	    "classification": "categorical_crossentropy",
    }
    lossWeights = {"embedding": 0.5, "classification": 0.5}

    mdl.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics=[CategoricalAccuracy())

        
    checkpoint_ = ModelCheckpoint(model_path, verbose=1, monitor='val_loss',save_best_only=True, mode='min')  
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=0.00000001, verbose=1)

    training_history = mdl.fit(x=X_train, y=[Y_train,Y_train_onehot], batch_size=256, epochs=epochs, validation_data=(X_val,[Y_val,Y_val_onehot]), callbacks=[ checkpoint_,reduce_lr], verbose=1)    

    if plot:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(training_history.history['loss'], 'b', label='Training Loss')
        plt.plot(training_history.history['val_loss'], 'r', label='Test Loss')
        plt.legend(loc='upper left')
        plt.title('Loss')
        plt.xlabel('Epochs')

        plt.subplot(1,2,2)
        plt.plot(training_history.history['categorical_accuracy'], 'b', label='Training Loss')
        plt.plot(training_history.history['val_categorical_accuracy'], 'r', label='Test Loss')
        plt.legend(loc='upper left')
        plt.title('Categorical Accuracy')
        plt.xlabel('Epochs')
        plt.show()

    return mdl, training_history
