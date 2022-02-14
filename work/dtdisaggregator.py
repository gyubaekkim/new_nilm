# The Improved Power Disaggregation (Non-intrusive Load Monitoring)
# Author : Gyubaek Kim (database.kim@yonsei.ac.kr)
# Description
#          Decision tree-based disaggregation models by our approach
# License
    # All rights reserved. No part of this code may be reproduced or transmitted in any form or by any means, or utilized by any information storage and 
    # retrieval system without written permission from the copyright owner. You can use this source code for free projects only. I will not allow 
    # commercial use of this code. I am not allowing anyone to modify any or all parts of the source code for commercial purposes. By using my source 
    # code, you agree to the following: 
    # 1. You will not distribute any or all parts of this source code for commercial use 
    # 2. You will cite me as the original creator of this source code. 
    # 3. You will inform me of its use before you use any or all parts of this source code. 
    # For use of any or all parts of this source code, email me. This source code is provided free to use for everybody provided this is used non-
    # commercially. Violators of this agreement will be subject to legal action by the author. If you see commercial applications or software using any or 
    # all parts of this source code, inform me.

from __future__ import print_function, division

import random
import sys
import math

from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import h5py

import pickle

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from nilmtk.legacy.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore
from nilmtk import DataSet

class DTDisaggregator(Disaggregator):
    
    def __init__(self, algorithm, use_feature, reactive):
        
        self.MODEL_NAME = "DT"
        self.algorithm = algorithm
        self.use_feature = use_feature
        self.reactive = reactive
        self.mmax = None
        self.meter_mmax = None
        self.MIN_CHUNK_LENGTH = 100
        self.n_trees = 100
        self.model = self.create_model(algorithm)

    def add_trees(self):
        self.n_trees = self.n_trees + 100
        self.model.set_params(n_estimators=self.n_trees)
        
    def train(self, mains, meter, epochs=1, batch_size=128, **load_kwargs):

        main_power_series = mains.power_series(**load_kwargs)
        meter_power_series = meter.power_series(**load_kwargs)           
     
        # Train chunks
        run = True
        
        mainchunk = next(main_power_series)
        meterchunk = next(meter_power_series)        
        
        # Reactive
        if self.reactive == True:
            main_reactive_series = mains.power_series(ac_type='reactive')
            reactivechunk = next(main_reactive_series)
        
        if self.mmax == None:
            self.mmax = mainchunk.max()
            self.meter_mmax = meterchunk.max()

        while(run):

            try: 
               
                mainchunk = self._normalize(mainchunk, self.mmax)
                meterchunk = self._normalize(meterchunk, self.mmax)
                
                # Reactive
                if self.reactive == True:
                    reactivechunk = self._normalize(reactivechunk, reactivechunk.max())

                # Reactive
                if self.reactive == True:
                    self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size, reactivechunk)
                else:
                    self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size)


                mainchunk = next(main_power_series)
                meterchunk = next(meter_power_series)
               
                # Reactive
                if self.reactive == True:
                    reactivechunk = next(main_reactive_series)
                
            except Exception as e:
                #print('Train is error..', e)
                run = False
                
    def train_on_chunk(self, mainchunk, meterchunk, epochs, batch_size, reactivechunk=None):
        
        # Replace NaNs with 0s because public datasets contain some NaNs
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True) 
        
        # Reactive
        if reactivechunk is not None:
            reactivechunk.fillna(0, inplace=True)
        
        ix = mainchunk.index.intersection(meterchunk.index)           
        #print(ix)
        
        mainchunk = np.array(mainchunk[ix])
        meterchunk = np.array(meterchunk[ix])
        
        mainchunk = np.reshape(mainchunk, (-1,1))

        # Reactive
        if reactivechunk is not None:
            reactivechunk = np.array(reactivechunk[ix])
       
        # Reactive
        if reactivechunk is not None:        
            reactivechunk = np.reshape(reactivechunk, (-1,1))
                
        if self.use_feature == True:    
            
            plt.plot(mainchunk)
            plt.plot(meterchunk)
            
            # Reactive
            if reactivechunk is not None:  
                plt.plot(abs(reactivechunk))
                plt.legend(('active', 'appliance', 'reactive'))
            else:
                plt.legend(('active', 'appliance'))              

            plt.title('Raw Data')
            plt.show()
        
            ## feature selection
            # Reactive
            if reactivechunk is not None:          
                feature_names, mainchunk = self._feature_extract(mainchunk, reactivechunk)
            else: 
                feature_names, mainchunk = self._feature_extract(mainchunk)
            print('feature selection is done..')

            self.model.fit(mainchunk, meterchunk)
            print('model fit is done..')

            important_features = pd.Series(data=self.model.feature_importances_,index=feature_names)
            important_features.sort_values(ascending=False,inplace=True)        
            print(important_features)
            important_features.nlargest(20).plot(kind='bar')
            important_features.plot(kind='bar')
            plt.show()
        
        else:
            
            self.model.fit(mainchunk, meterchunk)
        
        print('train on chunk is done..')

    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        
        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        #load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'     
        
        # Reactive
        if self.reactive == True:
            main_reactive_series = mains.power_series(ac_type='reactive')
            reactivechunk = next(main_reactive_series)
            reactivechunk = self._normalize(reactivechunk, reactivechunk.max())
       
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            chunk2 = self._normalize(chunk, self.mmax)
            
            # Reactive
            if self.reactive == True:
                appliance_power = self.disaggregate_chunk(chunk2, reactivechunk)
            else:
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
        
        print('Disaggregation is done..')
        
          
    def disaggregate_chunk(self, mains, reactivechunk=None):        

        up_limit = len(mains)

        mains.fillna(0, inplace=True)
        
        X_batch = np.array(mains)
        X_batch = np.reshape(X_batch, (-1,1))
        
        # Reactive
        if reactivechunk is not None:  
            reactivechunk.fillna(0, inplace=True)

        ix = mains.index
        
        # Reactive
        if reactivechunk is not None:  
            reactivechunk = np.array(reactivechunk[ix])
            reactivechunk = np.reshape(reactivechunk, (-1,1))
        
        if self.use_feature == True:
        
            ## feature selection
            if reactivechunk is not None:  
                feature_names, X_batch = self._feature_extract(X_batch, reactivechunk)    
            else:
                feature_names, X_batch = self._feature_extract(X_batch)     
            print('feature selection is done..')
            #print(X_batch)

        pred = self.model.predict(X_batch)
        pred = np.reshape(pred, (len(pred)))
        column = pd.Series(pred, index=mains.index[:len(X_batch)], name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        
        return appliance_powers    
    
    def _feature_extract(self, active, reactivechunk=None):
        
        feature_names = []

        if reactivechunk is None:
            feature_names = ['a0','a1','a2','a3','a4','a5','a6','a7','a8','a9',
                             'f0','f1','f2','f3','f4','f5','f6','f7','f8','f9',
                             'max','min','diff','mean','std','q0','q1','q2']
        else:
            feature_names = ['a0','a1','a2','a3','a4','a5','a6','a7','a8','a9',
                             'r0','r1','r2','r3','r4','r5','r6','r7','r8','r9',
                             'ap0','ap1','ap2','ap3','ap4','ap5','ap6','ap7','ap8','ap9',
                             'pf0','pf1','pf2','pf3','pf4','pf5','pf6','pf7','pf8','pf9',                             
                             'f0','f1','f2','f3','f4','f5','f6','f7','f8','f9',
                             'max','min','diff','mean','std','q0','q1','q2']            

        # reactivechunk may contain nan
        np.nan_to_num(reactivechunk, copy=False)
            
        n_features = len(feature_names)
        n_size = active.size
        
        result = np.empty((n_size, n_features))
        
        for i in range(n_size):

            try:                
                window = []
                val = []               
                
                a0 = active[0 if i < 9 else (i-9)][0]
                a1 = active[0 if i < 8 else (i-8)][0]
                a2 = active[0 if i < 7 else (i-7)][0]
                a3 = active[0 if i < 6 else (i-6)][0]
                a4 = active[0 if i < 5 else (i-5)][0]
                a5 = active[0 if i < 4 else (i-4)][0]
                a6 = active[0 if i < 3 else (i-3)][0]
                a7 = active[0 if i < 2 else (i-2)][0]
                a8 = active[0 if i < 1 else (i-1)][0]
                a9 = active[i][0]
               
                # window
                window.append(a0)
                window.append(a1)
                window.append(a2)
                window.append(a3)
                window.append(a4)
                window.append(a5)
                window.append(a6)
                window.append(a7)
                window.append(a8)
                window.append(a9)                
               
                # active
                val.append(window[0])
                val.append(window[1])
                val.append(window[2])
                val.append(window[3])
                val.append(window[4])
                val.append(window[5])
                val.append(window[6])
                val.append(window[7])
                val.append(window[8])
                val.append(window[9])
                
                if reactivechunk is not None:
                    
                    # reactive
                    r0 = reactivechunk[0 if i < 9 else (i-9)][0]
                    r1 = reactivechunk[0 if i < 8 else (i-8)][0]
                    r2 = reactivechunk[0 if i < 7 else (i-7)][0]
                    r3 = reactivechunk[0 if i < 6 else (i-6)][0]
                    r4 = reactivechunk[0 if i < 5 else (i-5)][0]
                    r5 = reactivechunk[0 if i < 4 else (i-4)][0]
                    r6 = reactivechunk[0 if i < 3 else (i-3)][0]
                    r7 = reactivechunk[0 if i < 2 else (i-2)][0]
                    r8 = reactivechunk[0 if i < 1 else (i-1)][0]
                    r9 = reactivechunk[i][0]

                    val.append(r0)
                    val.append(r1)
                    val.append(r2)
                    val.append(r3)
                    val.append(r4)
                    val.append(r5)
                    val.append(r6)
                    val.append(r7)
                    val.append(r8)
                    val.append(r9)

                    ap0 = np.sqrt(math.pow(a0,2)+math.pow(r0,2))
                    ap1 = np.sqrt(math.pow(a1,2)+math.pow(r1,2))
                    ap2 = np.sqrt(math.pow(a2,2)+math.pow(r2,2))
                    ap3 = np.sqrt(math.pow(a3,2)+math.pow(r3,2))
                    ap4 = np.sqrt(math.pow(a4,2)+math.pow(r4,2))
                    ap5 = np.sqrt(math.pow(a5,2)+math.pow(r5,2))
                    ap6 = np.sqrt(math.pow(a6,2)+math.pow(r6,2))
                    ap7 = np.sqrt(math.pow(a7,2)+math.pow(r7,2))
                    ap8 = np.sqrt(math.pow(a8,2)+math.pow(r8,2))
                    ap9 = np.sqrt(math.pow(a9,2)+math.pow(r9,2))

                    pf0 = (0 if ap0 == 0 else (a0 / ap0))
                    pf1 = (0 if ap1 == 0 else (a1 / ap1))
                    pf2 = (0 if ap2 == 0 else (a2 / ap2))
                    pf3 = (0 if ap3 == 0 else (a3 / ap3))
                    pf4 = (0 if ap4 == 0 else (a4 / ap4))
                    pf5 = (0 if ap5 == 0 else (a5 / ap5))
                    pf6 = (0 if ap6 == 0 else (a6 / ap6))
                    pf7 = (0 if ap7 == 0 else (a7 / ap7))
                    pf8 = (0 if ap8 == 0 else (a8 / ap8))
                    pf9 = (0 if ap9 == 0 else (a9 / ap9))

                    ## apparent
                    val.append(ap0)
                    val.append(ap1)
                    val.append(ap2)
                    val.append(ap3)
                    val.append(ap4)
                    val.append(ap5)
                    val.append(ap6)
                    val.append(ap7)
                    val.append(ap8)
                    val.append(ap9)

                    ## power factor
                    val.append(pf0)
                    val.append(pf1)
                    val.append(pf2)
                    val.append(pf3)
                    val.append(pf4)
                    val.append(pf5)
                    val.append(pf6)
                    val.append(pf7)
                    val.append(pf8)
                    val.append(pf9)
 
                # fourier transform result (same length of original data)
                f_data = np.array(np.fft.fft(window),dtype=float)     
                val.append(f_data[0])
                val.append(f_data[1])
                val.append(f_data[2])
                val.append(f_data[3])
                val.append(f_data[4])
                val.append(f_data[5])
                val.append(f_data[6])
                val.append(f_data[7])
                val.append(f_data[8])
                val.append(f_data[9])

                # statistics 
                val.append(max(window))
                val.append(min(window))
                val.append(max(window)-min(window))
                val.append(np.mean(window))
                val.append(np.std(window))
                val.append(np.nanquantile(window,0.25))
                val.append(np.nanquantile(window,0.5))
                val.append(np.nanquantile(window,0.75))
                
                result[i] = val
            
            except Exception as e:
                print('feature selection error', e)
                continue
        
        return feature_names, result

    def import_model(self, filename):
        
        self.model = pickle.load(open(filename, "rb"))

    def export_model(self, filename):
        
        pickle.dump(self.model, open(filename, "wb"))
     
    def _normalize(self, chunk, mmax):
        
        tchunk = chunk / mmax
            
        return tchunk

    def _denormalize(self, chunk, mmax):

        tchunk = chunk * mmax
            
        return tchunk   
    
    def create_model(self, algorithm):

        if algorithm == 'DT':
            model = DecisionTreeRegressor()
        if algorithm == 'DRF':
            model = RandomForestRegressor(verbose=1, n_jobs=32, warm_start=True)            
        if algorithm == 'GBM':
            model = GradientBoostingRegressor()

        return model    