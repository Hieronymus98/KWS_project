# 2021.02.16 TUE
import librosa # audio processing liabrary
import os # to manipulate file
import json
import numpy as np
import math
# import matplotlib.pyplot as plt
# import librosa.display
DATASET_PATH = "speech_dataset"
JSON_PATH = "data2.json"
SAMPLES_TO_CONSIDER = 1000 # 1 sec worth of sound


# n_mfcc : # of coefficient by extracting
# hop_length : # of frame 
# n_fft : # of sample for fast Fourier Transform
def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=512, hop_length=32):
    
    # data dictionary
    data = {
        "mappings": [], # mapping keywords("on", "off", ...) on numbers
        "labels": [], # target output we expect
        "MFCCs": [], # input of audio file's MFCC
        "files": [], # file name("speech_dataset/on/1.wav, ...")
    }
    intv_samp = math.floor(22050 * 0.285)
    step = math.floor(intv_samp / SAMPLES_TO_CONSIDER)
    print(intv_samp, step)
    # loop through all the sub dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # we need to ensure that we're not at root level
        if dirpath is not dataset_path:
            
            # update mapping
            label = dirpath.split("/")[-1] # speech_dataset/down -> [speech_dataset, down] -> down
            data["mappings"].append(label)
            print("\nProcessing: '{}'".format(label))
            
            # loop through all the filenames and extract MFCCs
            for f in filenames:
                
                # get file path
                file_path = os.path.join(dirpath, f)
                
                # load audio file
                y, sr = librosa.load(file_path, sr = 22050)  
                # y: audio time series(ex. array([-1.407e-03, -4.461e-04, ..., -3.042e-05,  1.277e-05], dtype=float32))
                # sr : sample rate
                # you can choose sample rate of audio file using librosa.load(sr=xxxxx) but default value is 22050(1 sec of sound sr)
                
                mark = y.argmax()
                
                st_ind = mark - intv_samp + 1 
                
                if st_ind < 0:
                    st_ind = 0
                
                for j in range(st_ind, mark):
                    if y[j] > (y[mark] * 0.2):
                        break
                
                if j+step*(SAMPLES_TO_CONSIDER-1) < len(y):
                    signal = y[j:j+step*(SAMPLES_TO_CONSIDER-1)+1:step]
                    # for k in range(0, SAMPLES_TO_CONSIDER):
                        # signal.append(y[j+step*k])
                else:
                    sstep = math.floor((len(y)-j) / SAMPLES_TO_CONSIDER)
                    if sstep < 1:
                        signal = y[-1000::1]
                    else:
                        signal = y[j:j+sstep*(SAMPLES_TO_CONSIDER-1)+1:sstep]
                
              
                # extract the MFCCs
                MFCCs = librosa.feature.mfcc(signal, sr=SAMPLES_TO_CONSIDER, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                # signal : audio time series(ex. array([-1.407e-03, -4.461e-04, ..., -3.042e-05,  1.277e-05], dtype=float32))
                # sr : sample rate(length of signal)
                # n_mfcc : number of MFCCs to return
                    
                # store data
                data["labels"].append(i-1)
                data["MFCCs"].append(MFCCs.T.tolist())
                data["files"].append(file_path)
                print("{}: {}.".format(file_path, i-1))
              
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)
