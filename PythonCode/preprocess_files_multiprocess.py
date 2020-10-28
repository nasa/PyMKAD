#!${HOMNE}/anaconda3/bin/python

# __________________________________________________________________________
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
#
#
'''
@author: Bryan Matthews KBRWyle
         Data Science Group
         NASA Ames Research Center

This code is designed to process gzipped csv files and preprocess using Symbolic Aggregate approXimation (SAX).
These files are stored in a SVMlight format file. Usage:
$>python preprocess_files_multiprocess.py config.json number_of_processes(optional)

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
import sys,os
import SAX
from multiprocessing import Process
from progress.bar import IncrementalBar
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Distributed worker for loading, partitioning, and computing statistics of flights.
def worker(filelist,config,thread_id):
    first_time=True
    continuous_params = np.atleast_1d(np.genfromtxt(config['params']['continuous'],delimiter="\n",comments="@",dtype=str))
    discrete_params =   np.atleast_1d(np.genfromtxt(config['params']['discrete'],delimiter="\n",comments="@",dtype=str))
    data_cube = {'continuous':np.zeros((filelist.shape[0], 81, continuous_params.shape[0]), dtype=float),
                'discrete':np.zeros((filelist.shape[0], 81, discrete_params.shape[0]), dtype=float),
                 'continuous_params':continuous_params,'discrete_params':discrete_params, 'filelist':[]}
    bar = IncrementalBar('Task '+str(100+thread_id)[1:]+': Partitioning Flights...', max=len(filelist))
    for i,f in enumerate(filelist):
        data_cube['filelist'].append(os.path.basename(f).split(".")[0])
        try:
            Data = SAX.load_FOQA_csv(f)
            markers=SAX.find_marker(Data,config['important_params'])
            Data=SAX.get_approach(Data,config['starting_alt'],markers)
        except ValueError:
            continue
        config['params']['continuous_indx']=SAX.find_param_indices(Data['header'],continuous_params)
        config['params']['discrete_indx']=SAX.find_param_indices(Data['header'],discrete_params)
        if(first_time):
            statistics=SAX.zscore_stream(Data['data'][:,config['params']['continuous_indx']])
            first_time=False
        else:
            statistics=SAX.zscore_stream(Data['data'][:,config['params']['continuous_indx']],statistics)
        xvec = np.flipud(np.cumsum(np.flipud(Data['data'][:, np.where(np.array(Data['header']) == config['important_params']['ground_speed'])[0]])) / 3600)
        bins = [np.intersect1d(np.where((xvec >= d)), np.where(xvec < (d + 0.25))) for d in np.linspace(0, 20 - 0.25, 80)] #Create 20 NM to 0 NM vector in 0.25 mile bins.
        bins.append(np.intersect1d(np.where(xvec >= 20.0), np.where(xvec < np.inf))) 

        data_cube['continuous'][i, :, :] = np.array([np.mean(Data['data'][b,:][:,config['params']['continuous_indx']],axis=0) if len(b)>0 else np.zeros((len(config['params']['continuous_indx'])),dtype=float)*np.nan for b in np.flipud(bins)])
        data_cube['discrete'][i, :, :] = np.array([np.mean(Data['data'][b, :][:, config['params']['discrete_indx']],axis=0) if len(b) > 0 else np.zeros((len(config['params']['discrete_indx'])), dtype=float) * np.nan for b in np.flipud(bins)])
        pickle.dump(Data,open(os.path.join(config['working_dir'],'data',os.path.basename(f).replace('.csv.gz','.pkl')),'wb'))
        bar.next()
    bar.finish()
    pickle.dump(statistics,open(os.path.join(config['working_dir'],'statistics_'+str(thread_id)+'.pkl'),'wb'))
    pickle.dump(data_cube, open(os.path.join(config['working_dir'], 'data_cube_' + str(thread_id) + '.pkl'), 'wb'))
    return()

# Distributed worker for applying SAX vectorization to flight data. 
def worker_SAX(filelist,config,statistics,thread_id):
    
    good_indx=np.zeros((len(filelist)),dtype=bool)
    continuous_params = np.atleast_1d(np.genfromtxt(config['params']['continuous'],delimiter="\n",comments="@",dtype=str))    
    discrete_params =   np.atleast_1d(np.genfromtxt(config['params']['discrete'],delimiter="\n",comments="@",dtype=str))
    bar = IncrementalBar('Task '+str(100+thread_id)[1:]+': Creating SAX Vector...', max=len(filelist))
    first_time=True
    for i,f in enumerate(filelist):
        Data=pickle.load(open(f,'rb'))
        config['params']['continuous_indx']=SAX.find_param_indices(Data['header'],continuous_params)
        config['params']['discrete_indx']=SAX.find_param_indices(Data['header'],discrete_params)
        Data['data'][:,config['params']['continuous_indx']]=(Data['data'][:,config['params']['continuous_indx']]-np.tile(statistics['dataMean'],[Data['data'].shape[0],1]))/np.tile(statistics['dataStd'],[Data['data'].shape[0],1])
        Data['data'][np.isnan(Data['data'])]=0
        quantized_data=SAX.quantize_time_series(Data,config['params'],config['alphabet'],config['window_size'])
        good_indx[i]=quantized_data.shape[0]!=0
        if(not good_indx[i]):
            continue
        discrete_seq=SAX.convert_disc_2_seq(Data,config['params'])
        if(first_time):
            first_time=False
            # if(os.path.dirname(config['svmlight_file'])!=""):
            print(os.path.dirname(config['svmlight_file']))
            os.makedirs(os.path.dirname(config['svmlight_file']), exist_ok=True)
            SAX.output_vector_SVMlight(config['svmlight_file']+'_'+str(100+thread_id)[1:],False,quantized_data,discrete_seq)
        else:
            SAX.output_vector_SVMlight(config['svmlight_file']+'_'+str(100+thread_id)[1:],True,quantized_data,discrete_seq)
        bar.next()
    bar.finish()
    np.savetxt(os.path.join(config['working_dir'],"filelist_in_svmlight_file_"+str(100+thread_id)[1:]+".txt"),np.array(filelist)[good_indx],fmt="%s")
    return()

def cat_files(filelist,output):
    fid_out = open(output,'w')
    for f in filelist:
        with open(f,'r') as fid:
            data = fid.read()
        fid_out.write(data)
    fid_out.close()

if __name__ == '__main__':
    
    if(len(sys.argv)<2):
        print("Usage:")
        print("$>python preprocess_files_multiprocess.py config.json number_of_processes(optional)")
        quit()

    skip_paritioning=False #For debugging purposes. Skipps the initial partitioning of flights when True.
    
    # Limit multi threading in scientific packagages like BLAS to 1 process to avoid conflict with our multiprocess preprocess steps.
    os.environ["OMP_NUM_THREADS"] = "1" 
    
    startT = time.time()
    config=json.load(open(sys.argv[1]))
    if(len(sys.argv)<3):
        number_of_processes=1.0
    else:
        number_of_processes=float(sys.argv[2])

    os.makedirs(os.path.join(config['working_dir'],'data'), exist_ok=True)
    if not skip_paritioning:
        print("Partitioning flights from "+str(config['starting_alt'])+ " ft to landing...")
        filelist=np.genfromtxt(config['filelist'],delimiter='\n',dtype=str)
        size_per_thread=np.ceil(float(filelist.shape[0])/number_of_processes)
        jobs=[]
        for i in range(int(number_of_processes)):
            p = Process(target=worker, args=(filelist[int((i)*size_per_thread):int(min(int((i+1)*size_per_thread),filelist.shape[0]))],config,i))
            jobs.append(p)
            p.start()
        while len(jobs) > 0:
            jobs = [job for job in jobs if job.is_alive()]
            time.sleep(1)
    
        statistics=pickle.load(open(os.path.join(config['working_dir'],'statistics_0.pkl'),'rb'))
        data_cube = pickle.load(open(os.path.join(config['working_dir'],'data_cube_0.pkl'), 'rb'))
        for i in range(1,int(number_of_processes)):
            statistics2=pickle.load(open(os.path.join(config['working_dir'],'statistics_'+str(i)+'.pkl'),'rb'))
            statistics=SAX.zscore_stream_merge(statistics,statistics2)
            data_cube_tmp = pickle.load(open(os.path.join(config['working_dir'], 'data_cube_' + str(i) + '.pkl'), 'rb'))
            data_cube['continuous'] = np.vstack((data_cube['continuous'],data_cube_tmp['continuous']))
            data_cube['discrete'] = np.vstack((data_cube['discrete'], data_cube_tmp['discrete']))
            data_cube['filelist'].extend(data_cube_tmp['filelist'])

        pickle.dump(statistics,open(os.path.join(config['working_dir'],'statistics.pkl'),'wb'))
        pickle.dump(data_cube, open(os.path.join(config['working_dir'],'data_cube.pkl'), 'wb'))
        for i in range(int(number_of_processes)):
            os.remove(os.path.join(config['working_dir'],'statistics_'+str(i)+'.pkl'))
            os.remove(os.path.join(config['working_dir'],'data_cube_' + str(i) + '.pkl'))
    
    first_time=True
    statistics=pickle.load(open(os.path.join(config['working_dir'],'statistics.pkl'),'rb'))
    filelist=np.array(sorted(list(set(glob(os.path.join(config['working_dir'],'data','*.pkl'))))))
    size_per_thread=np.ceil(float(filelist.shape[0])/number_of_processes)
    jobs=[]
    for i in range(int(number_of_processes)):
        p = Process(target=worker_SAX, args=(filelist[int((i)*size_per_thread):int(min(int((i+1)*size_per_thread),filelist.shape[0]))],config,statistics,i))
        jobs.append(p)
        p.start()
    while len(jobs) > 0:
        jobs = [job for job in jobs if job.is_alive()]
        time.sleep(1)
        
    filelist = sorted(glob(os.path.join(config['working_dir'],'filelist_in_svmlight_file_*')))
    cat_files(filelist,os.path.join(config['working_dir'],'filelist_in_svmlight_file.txt'))
    # os.system('cat '+config['working_dir']+'/filelist_in_svmlight_file_* > '+ config['working_dir']+'/filelist_in_svmlight_file.txt')
    [os.remove(f) for f in glob(config['working_dir']+"/filelist_in_svmlight_file_*")]
    filelist = sorted(glob(config['svmlight_file']+'_*'))
    cat_files(filelist,config['svmlight_file'])
    # os.system('cat '+config['svmlight_file']+'_* > '+config['svmlight_file'])
    [os.remove(f) for f in glob(config['svmlight_file']+"_*")]

    print("Runtime:" + str(time.time()-startT) + "Seconds")


