#!${HOMNE}/anaconda3/bin/python

#_________________________________________________________________________
# 
# Notices:
# 
# Copyright 2010, 2019 United States Government as represented by the Administrator of the National Aeronautics and
# Space Administration.  All Rights Reserved.
# 
# Disclaimers
# 
# No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED,
# IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM
# TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM
# FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION,
# IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
# ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE
# PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
# DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE,
# AND DISTRIBUTES IT "AS IS."
# 
# Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT,
# ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE
# RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY
# DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT
# SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL
# AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL
# BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
# 
# __________________________________________________________________________

'''
@author: Bryan Matthews KBRWyle
         Data Science Group
         NASA Ames Research Center

This code will take the report and generate visualization plots for each flight using the statistics derrived by
all the flights in the data set to determine 10-90th percentiles and binary state probabilities for each distance
to touchdown. Usage: 
$>python visualization.py config.json number_of_processes(optional)

Code Updated: 2019-03-08
'''

import numpy as np
import sys,os
import time
import pickle
import json
import matplotlib.pyplot as plt
from multiprocessing import Process


def worker(filelist,data_cube,MKAD_file,config,thread_id):
    
    ## Hard coded constants ##
    Ncols = 6.0
    discrete_fuzzy_threshold = 0.30 # determines when a normally off or on discrete is marked abnormal
    xvec = np.linspace(20,0,81)
    
    ptileData=np.percentile(data_cube['continuous'],[10,90],axis=0)
    avg_discrete = np.mean(data_cube['discrete'],axis=0)
    for i,a in enumerate(filelist):
        plot_idx = len(data_cube['continuous_params'])
        Nrows = np.ceil(len(data_cube['continuous_params'])/Ncols)
        
        flight_indx = np.where(a==np.array(data_cube['filelist']))[0][0]
        
        indx_sorted = np.argsort(MKAD_file[i,4:].astype(float))[::-1]
        scores = MKAD_file[i,4+indx_sorted].astype(float)
        indx_most_anomalous_params = indx_sorted[np.where(np.cumsum(scores/np.sum(scores))<0.5)]

        #Continuous
        fig, axs = plt.subplots(int(Ncols),int(Nrows),figsize=[12,19])
        fig.subplots_adjust(hspace=.5)
        plt.suptitle(a+" (Continuous Parameters)", fontsize=16)
 
        axs=axs.ravel()
        for pltIdx in np.arange(plot_idx):
            excursions_below = data_cube['continuous'][flight_indx,:,pltIdx] < ptileData[0,:,pltIdx]
            excursions_above = data_cube['continuous'][flight_indx,:,pltIdx] > ptileData[1,:,pltIdx]
            # plot time series variable 
            axs[pltIdx].plot(xvec,data_cube['continuous'][flight_indx,:,pltIdx],linewidth=2,label="flight data")
            axs[pltIdx].plot(xvec,ptileData[0,:,pltIdx],'k--',label="10/90 percentile")
            axs[pltIdx].plot(xvec,ptileData[1,:,pltIdx],'k--')
            axs[pltIdx].plot(xvec[excursions_below],data_cube['continuous'][flight_indx,excursions_below,pltIdx],'r.',markersize=10,linewidth=2,label="above|below percentile")
            axs[pltIdx].plot(xvec[excursions_above],data_cube['continuous'][flight_indx,excursions_above,pltIdx],'r.',markersize=10,linewidth=2)
            axs[pltIdx].invert_xaxis()
            if(pltIdx in indx_most_anomalous_params):
                axs[pltIdx].set_title("{}".format(data_cube['continuous_params'][pltIdx]),fontsize=10,color='red')
            else:
                axs[pltIdx].set_title("{}".format(data_cube['continuous_params'][pltIdx]),fontsize=10)
            axs[pltIdx].set_xlabel("Distance to Landing (NM)")
            if(pltIdx==1):
                axs[pltIdx].legend(loc=9, bbox_to_anchor=(0.5, 1.66), ncol=2)
        print("Saving:" + os.path.join(config['MKAD_folder'],'figs', a +'_c.pdf'))
        plt.savefig(os.path.join(config['MKAD_folder'],'figs', a +'_c.pdf'))
        plt.close()
        
        #Discretes
        plot_idx = len(data_cube['discrete_params'])
        Nrows = np.ceil(len(data_cube['discrete_params'])/Ncols)
        
        fig, axs = plt.subplots(int(Ncols),int(Nrows), figsize= [12,19])
        fig.subplots_adjust(hspace=.5)
        plt.suptitle(a+" (Discrete Parameters)", fontsize=16)
        axs=axs.ravel()
        for pltIdx in np.arange(plot_idx):
            
            high_probability_on = avg_discrete[:,pltIdx] > (0.5 + discrete_fuzzy_threshold)
            high_probability_off = avg_discrete[:,pltIdx] < (0.5 - discrete_fuzzy_threshold)
            
            excursions_off = (data_cube['discrete'][flight_indx,:,pltIdx] < (0.5 - discrete_fuzzy_threshold)) & high_probability_on
            excursions_on = (data_cube['discrete'][flight_indx,:,pltIdx] > (0.5 + discrete_fuzzy_threshold)) & high_probability_off
            
            # plot time series variable 
            axs[pltIdx].plot(xvec,(data_cube['discrete'][flight_indx,:,pltIdx]>0).astype(float),linewidth=2,label="flight data") #Have to threshold because we took the average over the 1/4 NM bin
            axs[pltIdx].plot(xvec,avg_discrete[:,pltIdx],'k--',label="average state")
            axs[pltIdx].plot(xvec[excursions_off],(data_cube['discrete'][flight_indx,excursions_off,pltIdx]>0).astype(float),'rs',markersize=8,linewidth=2,label="off when nominally on") #Have to threshold because we took the average over the 1/4 NM bin
            axs[pltIdx].plot(xvec[excursions_on],(data_cube['discrete'][flight_indx,excursions_on,pltIdx]>0).astype(float),'go',markersize=8,linewidth=2,label="on when nominally off") #Have to threshold because we took the average over the 1/4 NM bin
            axs[pltIdx].invert_xaxis()
            axs[pltIdx].set_title("{}".format(data_cube['discrete_params'][pltIdx]),fontsize=10)
            axs[pltIdx].set_xlabel("Distance to Landing (NM)")
            axs[pltIdx].set_ylim([-0.1,1.1])
            if(pltIdx==1):
                axs[pltIdx].legend(loc=9, bbox_to_anchor=(0.5, 1.66), ncol=2)
        print("Saving:" + os.path.join(config['MKAD_folder'],'figs' , a +'_d.pdf'))
        plt.savefig(os.path.join(config['MKAD_folder'],'figs' , a +'_d.pdf'))
        # plt.show()
        plt.close()
    """thread worker function to generate pdf plots"""
    print('Process '+str(thread_id) + ' done.')
    return()



if __name__ == '__main__':
    
    if(len(sys.argv)<2):
        print("Usage:")
        print("$>python visualization.py config.json number_of_processes(optional)")
        quit()
    
    config=json.load(open(sys.argv[1]))
    if(len(sys.argv)<3):
        number_of_processes=1.0
    else:
        number_of_processes=float(sys.argv[2])
    
    params_cont = np.genfromtxt(config['params']['continuous'],delimiter="\n",comments="@",dtype=str)
    params_disc = np.genfromtxt(config['params']['discrete'],delimiter="\n",dtype=str)
    
    filelist = np.genfromtxt(os.path.join(config['working_dir'],'filelist_in_svmlight_file.txt'),delimiter="\n",dtype=str)
    MKAD_file = np.genfromtxt(os.path.join(config['MKAD_folder'],'anomalous_flights_contributions_'+config['name']+'.csv'),delimiter=",",comments="@",dtype=str)[1:,:]

    
    anomaly_list = np.genfromtxt(os.path.join(config['MKAD_folder'],'anomalous_flights_contributions_'+config['name']+'.csv'),delimiter=",",comments="@",dtype=str)[1:,0]
    data_cube = pickle.load(open(os.path.join(config['working_dir'] , 'data_cube.pkl'),'rb'))

    root_good_filelist = [os.path.basename(f).replace('.pkl','') for f in filelist]

    good_indx = np.zeros((len(data_cube['filelist'])),dtype=bool)
    for i,a in enumerate(data_cube['filelist']):
        good_indx[i] = a in root_good_filelist

    data_cube['continuous'] = data_cube['continuous'][good_indx,:,:]
    data_cube['discrete'] = data_cube['discrete'][good_indx,:,:]
    data_cube['filelist'] = np.array(data_cube['filelist'])[good_indx]

    for i in range(data_cube['continuous'].shape[0]):
        if(np.sum(np.isnan(data_cube['continuous'][i,:,:]))>0):
            last_nan = np.max(np.where(np.isnan(data_cube['continuous'][i,:,0])==True))
            data_cube['continuous'][i,:last_nan+1,:] = data_cube['continuous'][i,last_nan+1,:]
            last_nan = np.max(np.where(np.isnan(data_cube['discrete'][i,:,0])==True))
            data_cube['discrete'][i,:last_nan+1,:] = data_cube['discrete'][i,last_nan+1,:]
   
    os.makedirs(os.path.join(config['MKAD_folder'],'figs'), exist_ok=True)
    # os.system('mkdir -p ' + config['MKAD_folder']+'/figs') 
    startT = time.time()
    
    size_per_thread=np.ceil(float(anomaly_list.shape[0])/number_of_processes)
    jobs=[]
    for i in range(int(number_of_processes)):
        p = Process(target=worker, args=(anomaly_list[int((i)*size_per_thread):int(min(int((i+1)*size_per_thread),anomaly_list.shape[0]))],data_cube,MKAD_file[int((i)*size_per_thread):int(min(int((i+1)*size_per_thread),anomaly_list.shape[0])),:],config,i))
        jobs.append(p)
        p.start()
    while len(jobs) > 0:
        jobs = [job for job in jobs if job.is_alive()]
        time.sleep(1)
    
    print("Runtime: " + str(time.time() - startT) + " Seconds")

        
        
        
