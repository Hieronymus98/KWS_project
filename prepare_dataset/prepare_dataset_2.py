# 2021.02.14 SUN
# zero padding
import librosa # audio processing liabrary
import os # to manipulate file
import json
import numpy as np

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
    lowest = 0
    highest = 0
    shortcnt = 0
    
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
                signal, sr = librosa.load(file_path, sr = SAMPLES_TO_CONSIDER)  
                # signal: audio time series(ex. array([-1.407e-03, -4.461e-04, ..., -3.042e-05,  1.277e-05], dtype=float32))
                # sr : sample rate
                # you can choose sample rate of audio file using librosa.load(sr=xxxxx) but default value is 22050(1 sec of sound sr)
                
                if len(signal) < SAMPLES_TO_CONSIDER:
                    shortcnt += 1
                    if shortcnt == 1:
                        lowest = len(signal)
                        highest = len(signal)
                    if lowest > len(signal):
                        lowest = len(signal)
                    if highest < len(signal):
                        highest = len(signal)
                        
                    x = []
                    for q in range(len(signal), SAMPLES_TO_CONSIDER):
                        x.append(0)
                    signal=np.append(signal, x)
                    print(len(signal))

                    
                # ensure the audio file is at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    
                    # enforce 1 sec, long signal
                    signal = signal[:SAMPLES_TO_CONSIDER]  
                    
                # extract the MFCCs
                MFCCs = librosa.feature.mfcc(signal, sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                # signal : audio time series(ex. array([-1.407e-03, -4.461e-04, ..., -3.042e-05,  1.277e-05], dtype=float32))
                # sr : sample rate(length of signal)
                # n_mfcc : number of MFCCs to return
                    
                # store data
                data["labels"].append(i-1)
                data["MFCCs"].append(MFCCs.T.tolist())
                data["files"].append(file_path)
                print("{}: {}.".format(file_path, i-1))
                print(lowest, highest)
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)
    
