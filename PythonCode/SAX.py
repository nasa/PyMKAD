'''
@author: Bryan Matthews KBRWyle
         Data Science Group
         NASA Ames Research Center

This code provides helper functions used by both preprocess_files_multiprocess.py and run_mkad.py. 

Code Updated: 2019-03-08
'''


import numpy as np
from scipy import signal
import csv
import json
import pickle
from glob import glob
import time
import gzip
import os,sys
import io
import pandas as pd
import nlcs

global cut_points
cut_points={ '2': [-np.inf,0],
             '3': [-np.inf,-0.43,0.43],
             '4': [-np.inf,-0.67,0,0.67],
             '5': [-np.inf,-0.84,-0.25,0.25,0.84],
             '6': [-np.inf,-0.97,-0.43,0,0.43,0.97],
             '7': [-np.inf,-1.07,-0.57,-0.18,0.18,0.57,1.07],
             '8': [-np.inf,-1.15,-0.67,-0.32,0,0.32,0.67,1.15],
             '9': [-np.inf,-1.22,-0.76,-0.43,-0.14,0.14,0.43,0.76,1.22],
             '10': [-np.inf,-1.28,-0.84,-0.52,-0.25,0,0.25,0.52,0.84,1.28],
             '11': [-np.inf,-1.34,-0.91,-0.6,-0.35,-0.11,0.11,0.35,0.6,0.91,1.34],
             '12': [-np.inf,-1.38,-0.97,-0.67,-0.43,-0.21,0,0.21,0.43,0.67,0.97,1.38],
             '13': [-np.inf,-1.43,-1.02,-0.74,-0.5,-0.29,-0.1,0.1,0.29,0.5,0.74,1.02,1.43],
             '14': [-np.inf,-1.47,-1.07,-0.79,-0.57,-0.37,-0.18,0,0.18,0.37,0.57,0.79,1.07,1.47],
             '15': [-np.inf,-1.5,-1.11,-0.84,-0.62,-0.43,-0.25,-0.08,0.08,0.25,0.43,0.62,0.84,1.11,1.5],
             '16': [-np.inf,-1.53,-1.15,-0.89,-0.67,-0.49,-0.32,-0.16,0,0.16,0.32,0.49,0.67,0.89,1.15,1.53],
             '17': [-np.inf,-1.56,-1.19,-0.93,-0.72,-0.54,-0.38,-0.22,-0.07,0.07,0.22,0.38,0.54,0.72,0.93,1.19,1.56],
             '18': [-np.inf,-1.59,-1.22,-0.97,-0.76,-0.59,-0.43,-0.28,-0.14,0,0.14,0.28,0.43,0.59,0.76,0.97,1.22,1.59],
             '19': [-np.inf,-1.62,-1.25,-1 -0.8,-0.63,-0.48,-0.34,-0.2,-0.07,0.07,0.2,0.34,0.48,0.63,0.8,1,1.25,1.62],
             '20': [-np.inf,-1.64,-1.28,-1.04,-0.84,-0.67,-0.52,-0.39,-0.25,-0.13,0,0.13,0.25,0.39,0.52,0.67,0.84,1.04,1.28,1.64]}

def read_pandas(filename):
    gz = gzip.open(filename, 'rb')
    f = io.BufferedReader(gz)
    data= pd.read_csv(f,low_memory=False).replace('False','0').replace('True','1').replace('DNE','nan')
    f.close()
    gz.close()
    header = np.array(data.keys())
    return(header,data.values.astype(float))


def quantize_lookup_table(x,alphabet_size):
    global cut_points
    return(alphabet_size-list(np.flipud(np.array(cut_points[str(alphabet_size)],dtype=float)<=x)).index(True))


def quantize_time_series(Data,params,alphabet,window_size):
    
    quantized_data = np.zeros((int(np.ceil(Data['data'].shape[0]/float(window_size))),len(params['continuous_indx'])),dtype=int)
    for i in range(int(np.ceil(Data['data'].shape[0]/float(window_size)))):
        jj=0
        for j in params['continuous_indx']:
            max_range=min([(i+1)*window_size-1,Data['data'].shape[0]])
            val = np.mean(Data['data'][i*window_size:max_range,j])
            quantized_data[i,jj]=quantize_lookup_table(val,alphabet)
            jj+=1
    return(quantized_data)

def convert_disc_2_seq(Data,params):
    if(len(params['discrete_indx'])>0):
        changes=np.diff(Data['data'][:,params['discrete_indx']],axis=0)
        for i in range(changes.shape[1]):
            changes[changes[:,i]==1,i]=(i+1)*2-1
            changes[changes[:,i]==-1,i]=(i+1)*2
        seq=changes.flatten()
        seq=np.append(seq[seq!=0],0)
    else:
        seq=np.array([1])
    return(seq)

def output_vector_SVMlight(filename,append,quantized_data,discrete_seq):
    FeatureV=[len(discrete_seq),quantized_data.shape[0]]
    FeatureV.extend(list(discrete_seq.astype(int)))
    FeatureV.extend(list(np.transpose(quantized_data).flatten()))
    if(append):
        fid=open(filename,'a')
    else:
        fid=open(filename,'w')
    fid.write("1 ")
    for i in range(len(FeatureV)):
        fid.write(str(i+1)+":"+str(FeatureV[i])+" ")
    fid.write("\n")
    fid.close()
    return([])


def find_param_indices(header,params):
    indx=[]
    for p in params:
        indx.append(list(header).index(p))
    return(tuple(indx))
    
def load_FOQA_csv(filename):
    header,data = read_pandas(filename) 
    data[0,np.isnan(data[0,:])]=0
    for i,row in enumerate(data[:-1,:]):
        indx_nans = np.isnan(data[i+1,:])
        data[i+1,indx_nans] = data[i,indx_nans]
    return({'header':header,'data':data})

#Finds touchdown point and decent beggining at cutoff altitude. 
def find_marker(Data,important_params):
    alt_indx=list(Data['header']).index(str(important_params['alt']))
    td_indicator_indx=list(Data['header']).index(str(important_params['td_indicator']))
    middle_indx=list(signal.filtfilt(np.ones((30),dtype=float),np.ones((1),dtype=float),Data['data'][:,alt_indx])/30**2>15000).index(True) #30 sec windowed filter to get rid of startup noise.
    td_indx=list(np.diff(Data['data'][middle_indx:,td_indicator_indx])>0).index(1)+middle_indx
    return({'middle_indx':middle_indx,'td_indx':td_indx,'alt_indx':alt_indx})

def get_approach(Data,start_alt,markers):
    ##Adjust altitudes by touchdown altitude##
    Data['data'][:,markers['alt_indx']]=Data['data'][:,markers['alt_indx']]-Data['data'][markers['td_indx'],markers['alt_indx']]
    start_indx=markers['td_indx']-list(np.flipud(Data['data'][markers['middle_indx']:markers['td_indx'],markers['alt_indx']])>start_alt).index(True)
    Data['data']=Data['data'][start_indx:markers['td_indx'],:]
    return(Data)

# Keeps track of first order statistics using Welford's Online algorithm
def zscore_stream(data,statistics={'dataMean':[],'dataStd':[],'S0':[],'S1':[],'S2':[]}):
    
    if(len(statistics['dataMean'])==0):
        statistics['S0']=np.zeros((data.shape[1]),dtype=int)
        statistics['S1']=np.zeros((data.shape[1]),dtype=int)
        statistics['S2']=np.zeros((data.shape[1]),dtype=int)
        statistics['dataMean']=np.zeros((data.shape[1]),dtype=float)
        statistics['dataStd']=np.zeros((data.shape[1]),dtype=float)
       
    statistics['S0']=statistics['S0']+np.sum(data**0,axis=0)
    statistics['S1']=statistics['S1']+np.sum(data**1,axis=0)
    statistics['S2']=statistics['S2']+np.sum(data**2,axis=0)
    
    for i in range(data.shape[1]):
        statistics['dataMean'][i]=statistics['S1'][i]/statistics['S0'][i]
        statistics['dataStd'][i]=(1.0/statistics['S0'][i])*np.sqrt(np.abs(statistics['S0'][i]*statistics['S2'][i]-statistics['S1'][i]**2))
    statistics['dataStd'][statistics['dataStd']==0]=1
    return(statistics)

# Merge reduce function to compute global statistics.
def zscore_stream_merge(statistics1,statistics2):
          
    statistics1['S0']+=statistics2['S0']
    statistics1['S1']+=statistics2['S1']
    statistics1['S2']+=statistics2['S2']
    
    for i in range(statistics1['dataMean'].shape[0]):
        statistics1['dataMean'][i]=statistics1['S1'][i]/statistics1['S0'][i]
        statistics1['dataStd'][i]=(1.0/statistics1['S0'][i])*np.sqrt(np.abs(statistics1['S0'][i]*statistics1['S2'][i]-statistics1['S1'][i]**2))
    statistics1['dataStd'][statistics1['dataStd']==0]=1
    return(statistics1)

# Calls nlcs from c-extensions code Compiled separately with kernels module
def MKAD_kernel_function(A,B):
    return(nlcs.compute(np.atleast_2d(np.array(A,dtype=np.uint16)),np.atleast_2d(np.array(B,dtype=np.uint16))))
