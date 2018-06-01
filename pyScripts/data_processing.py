# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:56:38 2018

@author: jayphate
"""
from scipy.io import wavfile
import librosa
import pandas as pd
import numpy as np
import os


def load_wav_file(fname, path):
    sr, b = wavfile.read(os.path.join(path,fname ))
    #assert sr == SAMPLE_RATE
    return sr, b, fname
    
def get_time_data(trainfile_path, audio_dir_path):
    
    train = pd.read_csv(trainfile_path)
    res = [ load_wav_file(fname, audio_dir_path) for fname in train.fname.values  if fname in os.listdir(dir1)]
    s, t, f = map(list, zip(*res))
    return s, t, f

def librosa_mfcc(fname, path, SAMPLE_RATE = 44100):
    wav, s_rate = librosa.core.load(os.path.join(path,fname ), sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(wav, sr = SAMPLE_RATE, n_mfcc=40)
    return mfcc,fname 

def get_mfcc(trainfile_path, audio_dir_path):
    
    train = pd.read_csv(trainfile_path)
    res = [ librosa_mfcc(fname, audio_dir_path) for fname in train.fname.values  if fname in os.listdir(dir1)]
    mfcc, fs = map(list, zip(*res))
    return mfcc, fs
    
    
    
#%%
 
trainfile_path = '../input/train.csv'
dir1 = '../audio_train_subset'
sample_rates, time_data, file_names = get_time_data(trainfile_path, dir1)
                   
mfcc_vector, fnames = get_mfcc(trainfile_path, dir1)

#%%




#%%


