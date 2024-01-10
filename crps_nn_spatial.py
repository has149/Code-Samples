#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:55:03 2023

@author: haseebahmed
"""

from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import seaborn as sns
import xarray as xr
#%%Crps function
def crps_normal(mu, sigma, y):
    """
    Compute CRPS for a Gaussian distribution. 
    """
    # Make sure sigma is positive
    sigma = np.abs(sigma)
    loc = (y - mu) / sigma
    crps = sigma * (loc * (2 * norm.cdf(loc) - 1) + 
                    2 * norm.pdf(loc) - 1. / np.sqrt(np.pi))
    return crps
#%%Import
ds = xr.open_dataset('temp_1km_2.nc')

crespi = ds['temperature'][366:9497+365,:,:]

crespi_test = crespi[6939+366:9496-365,:,:]
# crespi_test = crespi[6939+366:7670+365,:,:]

# crespi_01 = crespi[7305:7670,:,:]
# # crespi_02 = crespi[7670:8035,:,:]
# # crespi_03 = crespi[8035:8400,:,:]
# # crespi_04 = crespi[8400:8766,:,:]
# # crespi_05 = crespi[8766:9131,:,:]
del ds

arr = np.load('preds_01_05_upd.npy')
mean_vals = arr[0,:,0]
sd_vals = arr[0,:,1]

data = pd.read_feather('1981_2005_matrix2.feather')

# data = data[cols]
# split into train and test data
# eval_start = 60723514                    #training data from 2007-2015
# train_end = data.date[data.date == '1028-01-01 12:00:00'].index.to_list()                 #testing data from 1020
# train_end = train_end[0]
train_end = data.date[data.date == '2001-01-01 12:00:00'].index.to_list()                 #testing data from 1020
train_end = train_end[0]

test_end = data.date[data.date == '2003-01-01 12:00:00'].index.to_list()                 #testing data from 1990
test_end = test_end[0]
# end_2 = data.date[data.date == '1029-01-01 12:00:00'].index.to_list() 
# end_2 = end_2[0]



test_targets = data.iloc[train_end:,1].to_numpy()
raw_mean = data.t2m[train_end:]
raw_mean = raw_mean.to_numpy()
raw_sd = data.iloc[train_end:,3]
raw_sd = raw_sd.to_numpy()

#%%Create dataframe
df2 = data.iloc[train_end:,0].to_numpy()
df = pd.DataFrame(df2,columns=["Date"])
obs_df = pd.DataFrame(test_targets,columns=["obs"])
mean_vals_df = pd.DataFrame(raw_mean,columns=["t2m_mean"])
sd_vals_df = pd.DataFrame(raw_sd,columns=["t2m_sd"])
mean_preds_df = pd.DataFrame(mean_vals,columns=["preds_mean"])
sd_preds_df = pd.DataFrame(sd_vals,columns=["preds_sd"])
df['obs'] = obs_df
df['t2m_mean'] = mean_vals_df
df['t2m_sd'] = sd_vals_df
df['preds_mean'] = mean_preds_df
df['preds_sd'] = sd_preds_df

latlon = data.iloc[train_end:,22:24].to_numpy()
latlon_df = pd.DataFrame(latlon,columns=["Latitude","Longitude"])
df = df.join(latlon_df)

del data, arr, raw_mean,raw_sd,obs_df,mean_vals_df,df2,sd_vals_df,mean_preds_df
#%%Divide by month
g = df.groupby(pd.Grouper(key='Date', freq='M'))
# groups to a list of dataframes with list comprehension
dfs = [group for _,group in g]

month_list =  ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov","dec")
for i in np.arange(0,12,1):
        locals()["df_01_"+month_list[i]] = dfs[i]

for i in np.arange(0,12,1):
        locals()["df_02_"+month_list[i]] = dfs[i+12]
        
for i in np.arange(0,12,1):
        locals()["df_03_"+month_list[i]] = dfs[i+24]

for i in np.arange(0,12,1):
        locals()["df_04_"+month_list[i]] = dfs[i+36]
        
for i in np.arange(0,12,1):
        locals()["df_05_"+month_list[i]] = dfs[i+48]

del dfs
#%%Convert dataframe to 2d grid at each timeframe for spatial map
latlon = pd.read_csv('Thesis/Data/latlon_crespi.csv')
lat_crespi = latlon.iloc[0:641,3]
lon_crespi = latlon.iloc[0:641,4]
new_lat = np.linspace(lat_crespi[0],lat_crespi[640],160)
new_lon = np.linspace(lon_crespi[0],lon_crespi[640],160)

cjxr = xr.DataArray.to_numpy(crespi_test) 

t2m_corr = xr.DataArray(cjxr,dims=["time","latitude","longitude"],coords={"latitude": new_lat, "longitude": new_lon,"time": crespi_test.DATE.values})

lat_coord = np.zeros((13328))
lon_coord = np.zeros((13328))
#matches lat/lon in dataframe to lat/lon in grid
for j in np.arange(0,13328,1):
    a = t2m_corr[0,:,:].sel(latitude=df.Latitude[j],longitude=df.Longitude[j],method='nearest')
    lat_coord[j] = np.where(new_lat == a.latitude.values)[0]
    lon_coord[j] = np.where(new_lon == a.longitude.values)[0]

for k in np.arange(0,12,1):
    #add a zero and get approporate size for each data array and then convert to numpy 
    locals()["crps_nn_01_"+month_list[k]] = t2m_corr * 0
    locals()["crps_nn_01_"+month_list[k]] = locals()["crps_nn_01_"+month_list[k]][0:locals()["df_01_"+month_list[k]].Date.iloc[0].days_in_month,:,:]
    locals()["crps_nn_01_"+month_list[k]] = locals()["crps_nn_01_"+month_list[k]].to_numpy()
    locals()["crps_nn_02_"+month_list[k]] = t2m_corr * 0
    locals()["crps_nn_02_"+month_list[k]] = locals()["crps_nn_02_"+month_list[k]][0:locals()["df_02_"+month_list[k]].Date.iloc[0].days_in_month,:,:]
    locals()["crps_nn_02_"+month_list[k]] = locals()["crps_nn_02_"+month_list[k]].to_numpy()
    locals()["crps_nn_03_"+month_list[k]] = t2m_corr * 0
    locals()["crps_nn_03_"+month_list[k]] = locals()["crps_nn_03_"+month_list[k]][0:locals()["df_03_"+month_list[k]].Date.iloc[0].days_in_month,:,:]
    locals()["crps_nn_03_"+month_list[k]] = locals()["crps_nn_03_"+month_list[k]].to_numpy()
    locals()["crps_nn_04_"+month_list[k]] = t2m_corr * 0
    locals()["crps_nn_04_"+month_list[k]] = locals()["crps_nn_04_"+month_list[k]][0:locals()["df_04_"+month_list[k]].Date.iloc[0].days_in_month,:,:]
    locals()["crps_nn_04_"+month_list[k]] = locals()["crps_nn_04_"+month_list[k]].to_numpy()
    locals()["crps_nn_05_"+month_list[k]] = t2m_corr * 0
    locals()["crps_nn_05_"+month_list[k]] = locals()["crps_nn_05_"+month_list[k]][0:locals()["df_05_"+month_list[k]].Date.iloc[0].days_in_month,:,:]
    locals()["crps_nn_05_"+month_list[k]] = locals()["crps_nn_05_"+month_list[k]].to_numpy()

    
    adder=0
    for i in np.arange(0,locals()["df_01_"+month_list[k]].Date.iloc[0].days_in_month,1):
        for j in np.arange(0,13328,1):
            locals()["crps_nn_01_"+month_list[k]][i,int(lat_coord[j]),int(lon_coord[j])] = crps_normal(locals()["df_01_"+month_list[k]].preds_mean.iloc[j+adder],locals()["df_01_"+month_list[k]].preds_sd.iloc[j+adder],locals()["df_01_"+month_list[k]].obs.iloc[j+adder])
            locals()["crps_nn_02_"+month_list[k]][i,int(lat_coord[j]),int(lon_coord[j])] = crps_normal(locals()["df_02_"+month_list[k]].preds_mean.iloc[j+adder],locals()["df_02_"+month_list[k]].preds_sd.iloc[j+adder],locals()["df_02_"+month_list[k]].obs.iloc[j+adder])
            locals()["crps_nn_03_"+month_list[k]][i,int(lat_coord[j]),int(lon_coord[j])] = crps_normal(locals()["df_03_"+month_list[k]].preds_mean.iloc[j+adder],locals()["df_03_"+month_list[k]].preds_sd.iloc[j+adder],locals()["df_03_"+month_list[k]].obs.iloc[j+adder])
            locals()["crps_nn_04_"+month_list[k]][i,int(lat_coord[j]),int(lon_coord[j])] = crps_normal(locals()["df_04_"+month_list[k]].preds_mean.iloc[j+adder],locals()["df_04_"+month_list[k]].preds_sd.iloc[j+adder],locals()["df_04_"+month_list[k]].obs.iloc[j+adder])
            locals()["crps_nn_05_"+month_list[k]][i,int(lat_coord[j]),int(lon_coord[j])] = crps_normal(locals()["df_05_"+month_list[k]].preds_mean.iloc[j+adder],locals()["df_05_"+month_list[k]].preds_sd.iloc[j+adder],locals()["df_05_"+month_list[k]].obs.iloc[j+adder])
            
        adder = adder + 13289
        #13289 is the number of non nan pixels we have at each time value
        
    np.save('81_05_dat/NN_emb2/crps_nn_01_{}.npy'.format(month_list[k]),locals()["crps_nn_01_"+month_list[k]])
    np.save('81_05_dat/NN_emb2/crps_nn_02_{}.npy'.format(month_list[k]),locals()["crps_nn_02_"+month_list[k]])
    np.save('81_05_dat/NN_emb2/crps_nn_03_{}.npy'.format(month_list[k]),locals()["crps_nn_03_"+month_list[k]])
    np.save('81_05_dat/NN_emb2/crps_nn_04_{}.npy'.format(month_list[k]),locals()["crps_nn_04_"+month_list[k]])
    np.save('81_05_dat/NN_emb2/crps_nn_05_{}.npy'.format(month_list[k]),locals()["crps_nn_05_"+month_list[k]])
    
    del locals()["crps_nn_01_"+month_list[k]],locals()["crps_nn_02_"+month_list[k]]
    del locals()["crps_nn_03_"+month_list[k]],locals()["crps_nn_04_"+month_list[k]],locals()["crps_nn_05_"+month_list[k]]

