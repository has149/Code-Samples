#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 20:47:50 2023

@author: haseebahmed
"""

import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import percentileofscore
import time
import multiprocess as mp
import ctypes
import itertools

"""
EQM script that uses multiprocess to use multiple threads to speed up computation
Script was divided into even and odd numbered ensembles to save even more time
"""
#%%Obs and forecast datasets
ds = xr.open_dataset('temp_1km_2.nc')
# ds = xr.open_dataset('/Users/haseebahmed/Documents/Project/temp_1km.nc')

# crespi = ds['temperature'][366:4018,:,:]
crespi = ds['temperature'][366:9497+365,:,:]

crespi_test = crespi[6939+366:9496-365,:,:]

monthly_dat = crespi_test.groupby("DATE.month")

month_list =  ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov","dec")

crespi_np = crespi.to_numpy()
for i in np.arange(0,12,1):
    locals()["crespi_test_" + month_list[i]] = monthly_dat[i+1] * 0 

del ds

crespi_test_feb_1 = crespi_test_feb[0:112,:,:]
crespi_test_feb_2= crespi_test_feb[113:,:,:]
crespi_test_feb = xr.concat([crespi_test_feb_1,crespi_test_feb_2],dim="DATE")

#%%functions
def eQM_delta(obs_dataset, forecast_train, forecast_test):

        model_value = forecast_test[30]     #specific value from distribution
        percentile = percentileofscore(forecast_test, model_value) #percentile
        model_future_corrected = model_value + np.percentile(
            obs_dataset, percentile) - np.percentile(forecast_train, percentile)

        return model_future_corrected
    
def eqm_loop(params):
    """Main function that will be run in parallel"""
    idx = params[0]
    i = params[1]
    x = params[2]
    y = params[3]
    
    if np.isnan(crespi_np[0,x,y]) == True:
        corr_t2m_test[i,idx,x,y] = np.nan          
    else:
        corr_t2m_test[i,idx,x,y] = eQM_delta(mat_obs[i,:,:,x,y].flatten(),mat_for[i,:,:,x,y].flatten(),test_t2m[i,:,idx,x,y])


def to_numpy_array(shared_array, shape):
    '''Create a numpy array backed by a shared memory array.'''
    arr = np.ctypeslib.as_array(shared_array)
    return arr.reshape(shape)

def init_worker_rforreg(shared_array1,shape):
    '''
    Initialize worker for processing:
    Create the numpy array from the shared memory array for each process in the pool.
    '''
    global corr_t2m_test
    corr_t2m_test  = to_numpy_array(shared_array1, shape)
#%%Creating the datasets for EQM
#30 days ahead and 30 days behind for each day

#loop to go through all ensembles
for ens_num in np.arange(1,25,2):
   
    forecast_ens = np.load('81_05_dat/forecast_t2m_8106_ens.npy')
    forecast_t2m = forecast_ens[:,ens_num,:,:]  #goes through each ensemble here
    crespi0 = crespi*0
    
    forecast_t2m = forecast_t2m + crespi0
    
    del forecast_ens
    
    month_day = [31,28,31,30,31,30,31,31,30,31,30,31]
    month_num = ("01","02","03","04","05","06","07","08","09","10","11","12")
    
    #monthly loop
    for n in np.arange(0,12,1):
        mat_for = np.zeros((month_day[n],61,24,160,160))
        mat_obs = np.zeros((month_day[n],61,24,160,160))
        a = np.arange(1982,2006,1)

        for i in np.arange(1,month_day[n]+1,1):             #day of the month loop
           for y in np.arange(0,24,1):                      #year loop
               yr = forecast_t2m.indexes["DATE"].get_loc('{}-{}-01 00:00:00'.format(a[y],month_num[n]), method="nearest")
               #locate index in dataset of first day of the month for every year
               
               mat_for[i-1,:,y,:,:] = forecast_t2m[yr-31+i:yr+30+i,:,:].values
               mat_obs[i-1,:,y,:,:] = crespi[yr-31+i:yr+30+i,:,:].values
            
        test_t2m = mat_for[:,:,19:,:,:]
        test_obs = mat_obs[:,:,19:,:,:] + 273.15

        mat_for = mat_for[:,:,0:19,:,:]
        mat_obs = mat_obs[:,:,0:19,:,:] + 273.15

        # end = time.time()
        
        """setup for multiprocess pool (needs the definition of a separate shared
        array that can work with multiple threads)"""
        #ranges for the loops
        dim1, dim2, dim3, dim4 = np.arange(0,5,1),np.arange(0,month_day[n],1),np.arange(0,160,1),np.arange(0,160,1)
        paramlist = list(itertools.product(dim1,dim2,dim3,dim4))

        shape = (month_day[n],5,160,160)
        shared_array1 = mp.Array(ctypes.c_double, month_day[n]*5*160*160, lock=False)
        #processes controls the number of threads
        pool = mp.Pool(processes = 8,initializer=init_worker_rforreg,initargs=(shared_array1,shape))
                

       
        corr_t2m_test = to_numpy_array(shared_array1, shape)
        
        #this step runs the process in parallel
        pool.map(eqm_loop,paramlist)
                       
        pool.close()


        locals()["corr_t2m_test_"+month_list[n]] = corr_t2m_test
        locals()["corr_t2m_test2_"+month_list[n]] = np.zeros((month_day[n]*5,160,160))
        locals()["corr_t2m_test2_"+month_list[n]][0:month_day[n],:,:] =  locals()["corr_t2m_test_"+month_list[n]][:,0,:,:]
        locals()["corr_t2m_test2_"+month_list[n]][month_day[n]:month_day[n]*2,:,:] = locals()["corr_t2m_test_"+month_list[n]][:,1,:,:]
        locals()["corr_t2m_test2_"+month_list[n]][month_day[n]*2:month_day[n]*3,:,:] = locals()["corr_t2m_test_"+month_list[n]][:,2,:,:]
        locals()["corr_t2m_test2_"+month_list[n]][month_day[n]*3:month_day[n]*4,:,:] = locals()["corr_t2m_test_"+month_list[n]][:,3,:,:]
        locals()["corr_t2m_test2_"+month_list[n]][month_day[n]*4:month_day[n]*5,:,:] = locals()["corr_t2m_test_"+month_list[n]][:,4,:,:]
        locals()["corr_t2m_test2_"+month_list[n]] = locals()["corr_t2m_test2_"+month_list[n]] + locals()["crespi_test_" + month_list[n]]
        
        del test_t2m,test_obs,mat_for,mat_obs
        
    #combining all months and saving
    t2m_whole = xr.concat([corr_t2m_test2_jan,corr_t2m_test2_feb],dim="DATE")
    for idx in np.arange(2,12,1):
        t2m_whole = xr.concat([t2m_whole,locals()["corr_t2m_test2_"+month_list[idx]]],dim="DATE")

    t2m_whole = t2m_whole.sortby('DATE')
    
    
    np.save('81_05_dat/EQM_upd/t2m_corr_ens_{}.npy'.format(ens_num),t2m_whole)
    
    del t2m_whole
