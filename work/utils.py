# An Experimental Study on Performance of Non-intrusive Load Monitoring
# Author : Gyubaek Kim (database.kim@yonsei.ac.kr)
# Description
#          Utility functions to be used check similarity using dynamic time warping (DTW)
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

import pandas as pd
import numpy as np

import math
import itertools

from tslearn import metrics
from scipy import stats

from matplotlib import rcParams
import matplotlib.pyplot as plt

# data : list of dataframes, column : interesting columns
def showGraph(data, column):
    plt.figure(figsize=(10,4))
    plt.rcParams['axes.grid'] = True
    plt.xlabel ("Time")
    plt.ylabel ("Z Score")

    for i in range(len(data)):    
        for j in range(len(column)):
            plt.plot (range(len(data[i])), data[i][column[j]])
    
    plt.show ()
    
def checkDTW(data, column):
        
    if len(data) > 1:
        
        base = data[0][column[0]]
        
        # get the minimum length of item(dataframe) in the list
        min_length = min([len(i) for i in data])
        #print(min_length)        
        
        # baseline of distance
        acceptance = 0.3
        sum_of_acceptance = math.sqrt(len(base) * math.pow(np.amax(base) * acceptance, 2))
        display("Baseline = " + str(round(sum_of_acceptance,2)))
        
        distance_list = []
        
        for i in range(1, len(data)):
            
            try:
                
                query = data[i][column[0]]
                # calculate distance with same length dataframes
                distance = metrics.dtw(base[0:min_length], query[0:min_length]) 
                
                if math.isinf(distance):
                    continue
                    
                #display("DTW[0," + str(i) + "] : " + str(round(distance,2)))  
                distance_list.append(distance)
                
            except:
                print('DTW exception')
                continue
           
        
        display("[Average] : " + str(round((sum(distance_list)/len(distance_list)),2)))
        
    showGraph(data, column)
    
def checkSimilarity(data_list, always, **load_kwargs):
    
    df_list = []
    
    for idx in range(len(data_list)):
        data = data_list[idx]    
    
        df = data.to_frame(name='active')
        #print(df)    

        if always == True:            
            for i in range(len(df)):

                if i >= len(df)-1:
                    break

                if (i % 3600) == 0:
                    load = df[i:i+3600]
                    #print(load)
                    load['active'] = stats.zscore(load['active'])                
                    df_list.append(load)
        else:

            started = False
            start_idx = 0
            end_idx = 0

            for i in range(len(df)):

                if i == len(df)-1:
                    break

                if df.iloc[i]['active'] < 2:
                    if started == True:
                        started = False
                        end_idx = i                

                        load = df[start_idx:end_idx]
                        #print(load)
                        load['active'] = stats.zscore(load['active'])
                        df_list.append(load)
                    else:
                        start_idx = i
                else:
                    if started == False:
                        started = True
    
    checkDTW(df_list, ['active'])
    print('Done')
