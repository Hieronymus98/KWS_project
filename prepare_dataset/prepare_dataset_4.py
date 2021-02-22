# 2021.02.21 SUN
# train val data is sample rate of 22050Hz(converted to 1000Hz)
# but test data is natural 1000Hz
import librosa # audio processing liabrary
import os # to manipulate file
import json
import numpy as np
import math
import random
# import matplotlib.pyplot as plt
# import librosa.display
DATASET_PATH = "speech_dataset"
JSON_PATH_a = "data.json"
JSON_PATH_b = "t_data.json"
SAMPLES_TO_CONSIDER = 1000 # 1 sec worth of sound


# n_mfcc : # of coefficient by extracting
# hop_length : # of frame 
# n_fft : # of sample for fast Fourier Transform
def preprocess_dataset(dataset_path, json_path_a, json_path_b, num_mfcc=13, n_fft=512, hop_length=32):
    
    # data dictionary
    data = {
        "mappings": [], # mapping keywords("on", "off", ...) on numbers
        "labels": [], # target output we expect
        "MFCCs": [], # input of audio file's MFCC
        "files": [], # file name("speech_dataset/on/1.wav, ...")
    }
    
    test = {
        "mappings": [], # mapping keywords("on", "off", ...) on numbers
        "labels": [], # target output we expect
        "MFCCs": [], # input of audio file's MFCC
        "files": [], # file name("speech_dataset/on/1.wav, ...")
    }
    
    intv_samp = math.floor(22050 * 0.285)
    step = math.floor(intv_samp / SAMPLES_TO_CONSIDER)
    print(intv_samp, step)
    tot_cnt = 38546
    tst_cnt = 3855
    
    # loop through all the sub dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # we need to ensure that we're not at root level
        if dirpath is not dataset_path:
                
            # update mapping
            label = dirpath.split("/")[-1] # speech_dataset/down -> [speech_dataset, down] -> down
            print("\nProcessing: '{}'".format(label))
            
            # loop through all the filenames and extract MFCCs
            for f in filenames:
                # get file path
                file_path = os.path.join(dirpath, f)
                
                num = random.randint(0, 9)           
                if (tst_cnt > 0 and num == 0) or ((tst_cnt == tot_cnt) and tst_cnt >0):
                    tst_cnt -= 1
                    tot_cnt -= 1
                    y, sr = librosa.load(file_path, sr = 1000)
                    if len(signal) >= SAMPLES_TO_CONSIDER:
                    
                        signal = signal[:SAMPLES_TO_CONSIDER]  
                    
                        MFCCs = librosa.feature.mfcc(signal, sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    
                    test["mappings"].append(label)
                    test["labels"].append(i-1)
                    test["MFCCs"].append(MFCCs.T.tolist())
                    test["files"].append(file_path)
                    print("{}: {}.".format(file_path, i-1))
                
                else:    
                    tot_cnt -= 1
                    y, sr = librosa.load(file_path, sr = 22050)  
                
                    mark = y.argmax()
                
                    st_ind = mark - intv_samp + 1 
                
                    if st_ind < 0:
                        st_ind = 0
                
                    for j in range(st_ind, mark):
                        if y[j] > (y[mark] * 0.2):
                            break
                
                    if j+step*(SAMPLES_TO_CONSIDER-1) < len(y):
                        signal = y[j:j+step*(SAMPLES_TO_CONSIDER-1)+1:step]
                        
                    else:
                        sstep = math.floor((len(y)-j) / SAMPLES_TO_CONSIDER)
                        if sstep < 1:
                            signal = y[-1000::1]
                        else:
                            signal = y[j:j+sstep*(SAMPLES_TO_CONSIDER-1)+1:sstep]
                
                    MFCCs = librosa.feature.mfcc(signal, sr=SAMPLES_TO_CONSIDER, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                
                    
                    # store data
                    data["mappings"].append(label)
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                    print("{}: {}.".format(file_path, i-1))
              
    with open(json_path_a, "w") as fp:
        json.dump(data, fp, indent=4)
    
    with open(json_path_b, "w") as fp:
        json.dump(test, fp, indent=4)

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH_a, JSON_PATH_b)
