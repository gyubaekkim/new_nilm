from __future__ import print_function, division
import random
import sys

from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import h5py

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Reshape, Dropout
from keras.utils.vis_utils import plot_model

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.legacy.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

# ResNet
###################
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

class RESNETDisaggregator(Disaggregator):
###################

    def __init__(self):
        '''Initialize disaggregator

        Parameters
        ----------
        sequence_length : the size of window to use on the aggregate data
        meter : a nilmtk.ElecMeter meter of the appliance to be disaggregated
        '''
        
        # ResNet
        ####################
        sequence_length = 3600
        self.MODEL_NAME = "RESNET"
        ####################
        

        self.mmax = None
        self.sequence_length = sequence_length
        self.MIN_CHUNK_LENGTH = sequence_length

    def train(self, mains, meter, epochs=1, batch_size=16, **load_kwargs):
        '''Train

        Parameters
        ----------
        mains : a nilmtk.ElecMeter object for the aggregate data
        meter : a nilmtk.ElecMeter object for the meter data
        epochs : number of epochs to train
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''

        main_power_series = mains.power_series(**load_kwargs)
        meter_power_series = meter.power_series(**load_kwargs)

        # Train chunks
        run = True
        mainchunk = next(main_power_series)
        meterchunk = next(meter_power_series)
        if self.mmax == None:
            self.mmax = mainchunk.max()

        while(run):
            mainchunk = self._normalize(mainchunk, self.mmax)
            meterchunk = self._normalize(meterchunk, self.mmax)

            self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size)
            try:
                mainchunk = next(main_power_series)
                meterchunk = next(meter_power_series)
            except:
                run = False

    def train_on_chunk(self, mainchunk, meterchunk, epochs, batch_size):
        '''Train using only one chunk

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunk : chunk of appliance
        epochs : number of epochs for training
        '''

        s = self.sequence_length

        # Replace NaNs with 0s
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)
        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = mainchunk[ix]
        meterchunk = meterchunk[ix]

        # Create array of batches
        additional = s - (len(mainchunk) % s)
        X_batch = np.append(mainchunk, np.zeros(additional))
        Y_batch = np.append(meterchunk, np.zeros(additional))

        # ResNet    
        ################################
        
        X_batch = X_batch.reshape((-1, s, 1))
        X_batch = X_batch.reshape((X_batch.shape[0], X_batch.shape[1], 1, X_batch.shape[2])) 
        
        Y_batch = Y_batch.reshape((-1, s, 1))
        Y_batch = X_batch.reshape((Y_batch.shape[0], Y_batch.shape[1], 1, Y_batch.shape[2])) 
        
        self.model = self._create_model(64, X_batch.shape[1:])          
        #################################
        
        self.model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=epochs, shuffle=True)
        

    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        '''Disaggregate mains according to the model learnt.

        Parameters
        ----------
        mains : nilmtk.ElecMeter
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        meter_metadata : metadata for the produced output
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''

        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            chunk2 = self._normalize(chunk, self.mmax)

            appliance_power = self.disaggregate_chunk(chunk2)
            appliance_power[appliance_power < 0] = 0
            appliance_power = self._denormalize(appliance_power, self.mmax)

            # Append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

            # Append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

        # Save metadata to output
        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter_metadata]
            )

    def disaggregate_chunk(self, mains):
        '''In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series to disaggregate
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        '''
       
        s = self.sequence_length
        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        additional = s - (up_limit % s)
        X_batch = np.append(mains, np.zeros(additional))

        # ResNet        
        #####################################

        X_batch = X_batch.reshape((-1, s, 1))
        X_batch = X_batch.reshape((X_batch.shape[0], X_batch.shape[1], 1, X_batch.shape[2]))       
        
        pred = self.model.predict(X_batch).flatten()
        
        #####################################      
       
        
        pred = np.reshape(pred, (up_limit + additional))[:up_limit]
        column = pd.Series(pred, index=mains.index, name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers


    def import_model(self, filename):
        '''Loads keras model from h5

        Parameters
        ----------
        filename : filename for .h5 file

        Returns: Keras model
        '''
        self.model = load_model(filename)
        with h5py.File(filename, 'a') as hf:
            ds = hf.get('disaggregator-data').get('mmax')
            self.mmax = np.array(ds)[0]

    def export_model(self, filename):
        '''Saves keras model to h5

        Parameters
        ----------
        filename : filename for .h5 file
        '''
        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('mmax', data = [self.mmax])

    def _normalize(self, chunk, mmax):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''
        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):
        '''Deormalizes timeseries
        Note: This is not entirely correct

        Parameters
        ----------
        chunk : the timeseries to denormalize
        max : max value used for normalization

        Returns: Denormalized timeseries
        '''
        tchunk = chunk * mmax
        return tchunk

    # ResNet
    ##########################
    def _create_model(self, n_nodes, input_shape):

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1
        conv_x = keras.layers.BatchNormalization()(input_layer)
        conv_x = keras.layers.Conv2D(n_nodes, 8, 1, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(n_nodes, 5, 1, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(n_nodes, 3, 1, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv2D(n_nodes, 1, 1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.Add()([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2
        conv_x = keras.layers.Conv2D(n_nodes * 2, 8, 1, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(n_nodes * 2, 5, 1, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(n_nodes * 2, 3, 1, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv2D(n_nodes * 2, 1, 1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.Add()([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)
        
        # BLOCK 3
        conv_x = keras.layers.Conv2D(int(n_nodes * 2), 8, 1, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(int(n_nodes * 2), 5, 1, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(int(n_nodes * 2), 3, 1, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.Add()([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # ResNet
        ##################################
        # FINAL
        #full = keras.layers.GlobalAveragePooling2D()(output_block_3)
        ##Add dense layer with 32/64 nodes

        #output_layer = keras.layers.Dense(1)(full)
        output_layer = keras.layers.Dense(1)(output_block_3)
        
        ##################################
        
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')

        return model
    
    ##########################    