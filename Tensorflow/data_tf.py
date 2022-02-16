import numpy as np
import tensorflow as tf
import math

class DataGenerator(tf.keras.utils.Sequence):
    '''
    Generates data for tf.keras
    key params:
    list_IDs: prefix of the data in the disk.
    dim: dimension of the image.
    default: two-dimensional parameters, where I rescale the first parameter by a factor of 1/1.431
             data form: images, files with list of the names in '/scratch/zxs/all17/', e.g. Idlt-0.npy, Idlt-1.npy for training; 
                        Idlv-0.npy, Idlv-1.npy for validation; Idlp-0.npy, Idlp-1.npy for testing/prediction.
                        labels, files with list of the names in '/scratch/zxs/allo17/', e.g. Idlt-0-y.npy, Idlt-1-y.npy for training; 
                        Idlv-0-y.npy, Idlv-1-y.npy for validation; Idlp-0-y.npy, Idlp-1-y.npy for testing/prediction.
    
    '''
    def __init__(self, list_IDs, batch_size=32, dim=[66,66,660], n_channels=1,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size,self.dim[0],self.dim[1],self.dim[2],self.n_channels)) #image
        y = np.empty((self.batch_size,2)) #parameter

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,:,:,:,0] = np.load('/scratch/zxs/all17/' + ID + '.npy')
            y[i,] = np.load('/scratch/zxs/allo17/' + ID + '-y' + '.npy')
            y[i,0] = y[i][0]/1.431
        y1=y[:,0]
        y2=y[:,1]
        y=[y1,y2]

        return X, y


