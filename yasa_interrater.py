# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:09:40 2024

@author: WS3
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:39:32 2023

@author: adla

AUTOMATIC SLEEP SCORING
"""

#%% Loading modules
import yasa
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

#%% Loading the raw data

#selecting EEG file of session 1 of all subjects
# Parent path
root_dir_eegfile = r"/serverdata/ccshome/adla/NAS/CCS_SleepScorers/Dozee_sleepdata/Dozee Raw data/"

# Checking the files
for path in glob.glob(f'{root_dir_eegfile}/*/*/*_0001.edf', recursive=True):
    print("File Name: ", path.split('\\')[-1])
    print("Path: ", path)

# Files in a list
eegfiles = glob.glob(f'{root_dir_eegfile}/*/*/*')

    
filepath = glob.glob(f'{root_dir_eegfile}/*/*/*_0001.edf')


common = set(eegfiles) & set(filepath)
eegfiles = [i for i in eegfiles if i not in common]

#%% Sleep scoring 

accuracies1 = []
for i in range(len(eegfiles)):
    try:
        raw = mne.io.read_raw_edf(eegfiles[i], preload = False, verbose = True)
        fname = os.path.basename(eegfiles[i])[:-4]
        
        hypno = np.loadtxt(f'{fname}_0001_reduced_krishan.csv',dtype = str)
        sls = yasa.SleepStaging(raw, eeg_name = "EEG CZ-Ref") 
        y_pred = sls.predict()
        #y_pred
        accuracy = (hypno[0:len(y_pred)] == y_pred).sum() / y_pred.size
        accuracies1.append(accuracy)
        print(i, "/", len(eegfiles))
        print(accuracy)
    except: 
       print("An error occured")
       
       
       
np.mean(accuracies) * 100      
np.mean(accuracies1) * 100     

#%% Yasa hypnogram

for i in range(len(eegfiles)):
    raw = mne.io.read_raw_edf(eegfiles[i] , preload = False, verbose = True)
    fname = os.path.basename(eegfiles[i])[:-4]
    sls = yasa.SleepStaging(raw, eeg_name = "EEG CZ-Ref")
    y_pred = sls.predict()
    
    # converting W and R into 0 and 4 | In case any renaming is needed
    y_pred = [s.replace('W', '0') for s in y_pred]
    y_pred = [s.replace('R', '4') for s in y_pred]
    y_pred = [s.replace('N1', '1') for s in y_pred]
    y_pred = [s.replace('N2', '2') for s in y_pred]
    y_pred = [s.replace('N3', '3') for s in y_pred]
    
    
    plt.figure(figsize = (25, 2.5))
    yasa.plot_hypnogram(y_pred,
                        lw = 1)
    plt.tight_layout()
    plt.savefig(fname + '_yasahypno.png',
                dpi = 600)
    plt.close()