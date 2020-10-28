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

This code will load the SVMlight file produced by preprocess_files_multiprocess.py and execute the Multiple Kernel Anomaly
Detection (MKAD) algorithm. The output will be saved in a csv file with decomposed score compositions. Usage: 
$>python run_mkad.py config.json number_of_processes(optional)

Code Updated: 2019-03-08
'''



import sys,os
import json
import numpy as np
from multiprocessing import Process, Queue
import time
from sklearn.datasets import load_svmlight_file
import SAX
from progress.bar import IncrementalBar
from sklearn.svm import OneClassSVM
import pickle
from sklearn.cluster import DBSCAN


def parse_SAX_vector(SAX_v):
    seq = SAX_v[0,2:2+int(SAX_v[0,0])]
    num_rows = int(SAX_v[0,1])
    num_cols = int((SAX_v.shape[1]-int(SAX_v[0,0])-2)/int(SAX_v[0,1]))
    cont_matrix = SAX_v[0,2+int(SAX_v[0,0]):].reshape((num_cols,num_rows))
    return([seq,cont_matrix])

def worker(index,svmlight_data,thread_id,q):
    
    bar = IncrementalBar('Task '+str(100+thread_id)[1:]+': Computing Kernel...', max=len(index))
    K = np.zeros((len(index),svmlight_data.shape[0]),dtype=float)
    count = 0
    for I,i in enumerate(index):
        seq1,cont_matrix1 = parse_SAX_vector(svmlight_data[i,:svmlight_data.getrow(i).nonzero()[1][-1]+1].todense())
        for j in range(i,svmlight_data.shape[0]):
            seq2,cont_matrix2 = parse_SAX_vector(svmlight_data[j,:svmlight_data.getrow(j).nonzero()[1][-1]+1].todense())
            K[I,j] = 0.5*SAX.MKAD_kernel_function(np.transpose(seq1),np.transpose(seq2))
            for l in range(cont_matrix1.shape[0]):
                K[I,j] += 0.5*SAX.MKAD_kernel_function(np.transpose(cont_matrix1[l,:]),np.transpose(cont_matrix2[l,:]))/cont_matrix1.shape[0]
            count += 1
        bar.next()
    bar.finish()
    q.put(K)
    return([])

def worker_test(alphas,SVs,test,thread_id,q):
    
    _,cont_matrix = parse_SAX_vector(SVs[0,:np.max(SVs[0,:].nonzero()[1])+1].todense())
    num_contin = cont_matrix.shape[0]
    bar = IncrementalBar('Task '+str(100+thread_id)[1:]+': Calculating Decomposed Scores...', max=test.shape[0])
    
    scores_decomposed = np.zeros((test.shape[0],1+num_contin),dtype=float)
    for j in range(test.shape[0]):
        seq2,cont_matrix2 = parse_SAX_vector(test[j,:np.max(test[j,:].nonzero()[1])+1].todense())
        for i in range(SVs.shape[0]):
            seq1,cont_matrix1 = parse_SAX_vector(SVs[i,:np.max(SVs[i,:].nonzero()[1])+1].todense())
            scores_decomposed[j,0] += alphas[i]*SAX.MKAD_kernel_function(np.transpose(seq1),np.transpose(seq2))
            for l in range(num_contin):
                scores_decomposed[j,1+l] += alphas[i]*SAX.MKAD_kernel_function(np.transpose(cont_matrix1[l,:]),np.transpose(cont_matrix2[l,:]))
        bar.next()
    bar.finish()
    q.put(scores_decomposed)
    return([])


if __name__ == '__main__':
    
    if(len(sys.argv)<2):
        print("Usage:")
        print("$>python run_mkad.py config.json number_of_processes(optional)")
        quit()
        
    if(len(sys.argv)<3):
        number_of_processes=1.0
    else:
        number_of_processes=float(sys.argv[2])
        
    config=json.load(open(sys.argv[1]))
    
    startT = time.time()
    
    svmlight_data = load_svmlight_file(config['svmlight_file'])[0][:,:]
    nu = config['nu']
    working_dir = config['working_dir']
    params_c = np.genfromtxt(config['params']['continuous'],delimiter="\n",dtype=str)
    
    # Check to make sure kernel file exists. If not resets to compute kernel from SVMlight file and save kernel. 
    if(not os.path.isfile(os.path.join(config['working_dir'],'kernel_'+config['name']+'.pkl'))):
        print("No exisiting kernel found...Computing from SVMlightFile")
        config['use_existing_kernel'] = False
        config['save_kernel'] = True
    
    os.system('mkdir -p '+config['MKAD_folder'])
    if(not config['use_existing_kernel']):
        totals = np.cumsum(np.arange(svmlight_data.shape[0],1,-1))
        chunk_size = int(totals[-1]/number_of_processes)
        index = [0]
        while np.sum(totals) > 0:
            I = np.argmax(totals>chunk_size)
            if(I==0):
                index.append(totals.shape[0]+1)
                break
            index.append(I)
            totals -= totals[index[-1]]
            totals[:index[-1]] = 0
        
        size_per_thread=np.ceil(float(svmlight_data.shape[0])/number_of_processes)
        jobs=[]
        pipe_list = []
        for i in range(int(number_of_processes)):
            if(index[i]==svmlight_data.shape[0]):
                break
            q = Queue()
            p = Process(target=worker, args=(np.arange(index[i],index[i+1]),svmlight_data,i,q))
            jobs.append(p)
            pipe_list.append(q)
            p.start()
        
        time.sleep(1)
        
        K = np.zeros((svmlight_data.shape[0],svmlight_data.shape[0]),dtype=float)
        indx = 0
        for i,x in enumerate(pipe_list):
            tmp = x.get()
            K[indx:indx+tmp.shape[0],:] = tmp
            indx += tmp.shape[0]
        
        # Copy over the upper to lower triangle
        i_lower = np.tril_indices(K.shape[0],-1)
        K[i_lower] = np.transpose(K)[i_lower] #Keep consisten row major indexing by transposing and getting the upper.
        
        if(config['save_kernel']):
            pickle.dump(K,open(os.path.join(config['working_dir'],'kernel_'+config['name']+'.pkl'),'wb'))
    if(config['use_existing_kernel']):
        print("Loading Exisiting Kernel...")
        K=pickle.load(open(os.path.join(config['working_dir'],'kernel_'+config['name']+'.pkl'),'rb'))
    
    # Solve the one-class SVM
    clf = OneClassSVM(kernel='precomputed',nu=0.1,tol=1e-12)
    clf.fit(K)
    scores = clf.score_samples(K) - clf.offset_

    filelist = np.genfromtxt(working_dir+"/filelist_in_svmlight_file.txt",delimiter="\n",dtype=str)
    filelist = np.array([os.path.basename(f).split(".")[0] for f in filelist])
    
    sorted_indx = np.argsort(scores)
    cutoff_point = np.argmax(scores[sorted_indx]>=0)
    
    # Reduce scores and flights to anomaly list
    filelist_anoms = filelist[sorted_indx][:cutoff_point]
    scores = scores[sorted_indx][:cutoff_point]
    
    # Select data for Support Vectors and anomalies
    SVs = svmlight_data[clf.support_,:]
    anoms = svmlight_data[sorted_indx,:][:cutoff_point,:]
    del(K)
    
    # Normalize alphas to sum to 1
    alphas = clf.dual_coef_[0]/np.sum(clf.dual_coef_[0])
    
    # Get unbounded Support Vectors (used for computing rho)
    SVs_ub =  SVs[alphas <= 1/(clf.dual_coef_[0]*svmlight_data.shape[0]),:]

    _,cont_matrix1 = parse_SAX_vector(svmlight_data[0,:np.max(svmlight_data[0,:].nonzero()[1])+1].todense()) #get the number of continuous parameters.
    num_contin = cont_matrix1.shape[0]
    
    
    print("\nComputing Decomposed Rho Values...")
    # Decompose the rhos
    rho = np.zeros((1+num_contin),dtype=float)
    for i in range(SVs.shape[0]):
        seq1,cont_matrix1 = parse_SAX_vector(SVs[i,:np.max(SVs[i,:].nonzero()[1])+1].todense())
        for j in range(SVs_ub.shape[0]):
            seq2,cont_matrix2 = parse_SAX_vector(SVs_ub[j,:np.max(SVs_ub[j,:].nonzero()[1])+1].todense())
            rho[0] += alphas[i]*SAX.MKAD_kernel_function(np.transpose(seq1),np.transpose(seq2))
            for l in range(num_contin):
                rho[1+l] += alphas[i]*SAX.MKAD_kernel_function(np.transpose(cont_matrix1[l,:]),np.transpose(cont_matrix2[l,:]))#/cont_matrix1.shape[0]
    rho /= SVs_ub.shape[0]
 
       
    global_rho = np.sum(rho[1:]*0.5/num_contin)+rho[0]*0.5
    print(global_rho)
    
    print("Decomposing Scores for "+str(anoms.shape[0])+ " Anomalies...")
    size_per_thread=int(np.ceil(float(anoms.shape[0])/number_of_processes))
    jobs=[]
    pipe_list = []
    for i in range(int(number_of_processes)):
        q = Queue()
        p = Process(target=worker_test, args=(alphas,SVs,anoms[int(i)*size_per_thread:int(min(int((i+1)*size_per_thread),anoms.shape[0])),:],i,q))
        jobs.append(p)
        p.start()
        pipe_list.append(q)
    
    scores_decomposed = np.zeros((anoms.shape[0],1+num_contin),dtype=float)
    indx = 0
    for x in pipe_list:
        tmp = x.get()
        scores_decomposed[indx:indx+tmp.shape[0],:] = tmp
        indx += tmp.shape[0]
    
    print("Computing Contributions...")
    # Account for kernel weights and subtract out the decomposed rhos
    scores_decomposed[:,0] -= rho[0]
    scores_decomposed[:,0] *= 0.5
    for l in range(num_contin):
        scores_decomposed[:,1+l] -= rho[1+l]
        scores_decomposed[:,1+l] *= 0.5/num_contin
    
    # Compute the global scores using the normalized alphas 
    global_scores = np.sum(scores_decomposed,axis=1)- global_rho
    
    # Compute the percent contribution. 
    percent_contribution = np.zeros((anoms.shape[0],1+num_contin),dtype=float)
    for i,s in enumerate(scores_decomposed):
        percent_contribution[i,:] = (s-np.max(s))/np.sum(s-np.max(s))
    
    print("Clustering flights with similar contributions...")
    db = DBSCAN(eps=config['cluster_eps']).fit(percent_contribution)
    print(set(db.labels_))
    print("Number of Clusters: " + str(len(set(db.labels_))))
    
    print("Saving contribution file...\n"+config['MKAD_folder']+'/anomalous_flights_contributions_'+config['name']+'.csv')
    fid=open(config['MKAD_folder']+'/anomalous_flights_contributions_'+config['name']+'.csv','w')
    fid.write('Flight,MKAD_score,Cluster_ID,discrete_contribution,')
    fid.write(",".join(params_c)+"\n")
    for i in range(percent_contribution.shape[0]):
        fid.write(filelist_anoms[i]+","+str(round(global_scores[i],6))+','+str(db.labels_[i])+",")
        np.savetxt(fid,np.expand_dims(percent_contribution[i,:],axis=0),delimiter=",",fmt="%.6f")
    fid.close()
    print("Runtime:" + str(time.time()-startT) + "Seconds")