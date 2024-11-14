from librosa import time_to_frames,frames_to_time
from librosa.onset import onset_backtrack,onset_detect
# from essentia.standard import Windowing,FFT,CartesianToPolar,FrameGenerator,Onsets,OnsetDetection
# import essentia 
import numpy as np
from typing import *

def detect_onsets(audio, sr,with_backtrack):
    onsets = onset_detect(y=audio,sr=sr,backtrack=False,units='time')
    if with_backtrack:
        backtrack = onset_detect(y=audio,sr=sr,backtrack=True,units='time')
        
        return onsets, backtrack
    return onsets

# def detect_onsets(track, sampling_rate, with_backtrack) -> Tuple[np.ndarray,np.ndarray]:
#     if sampling_rate<44100 : raise ValueError("The sampling rate for essentia onset detect is otpimized for 44.1kHz. For lower rates use librosa.")
    
#     od_complex = OnsetDetection(method='complex')

#     w = Windowing(type='hann')
#     fft = FFT() # Outputs a complex FFT vector.
#     c2p = CartesianToPolar() # Converts it into a pair of magnitude and phase vectors.

#     pool = essentia.Pool()
#     for frame in FrameGenerator(track, frameSize=1024, hopSize=512):
#         magnitude, phase = c2p(fft(w(frame)))
#         pool.add('odf.complex', od_complex(magnitude, phase))

#     # 2. Detect onset locations.
#     onsets = Onsets()
#     onsets_complex = onsets(essentia.array([pool['odf.complex']]), [1])
    
#     #3 post process onsets : if any onset detected after duration -> remove
#     onsets_complex = onsets_complex[onsets_complex<len(track)/sampling_rate]
    
#     if not with_backtrack:
#         return onsets_complex

#     #TODO : UNIFIER DETECTION ONSET ET BACKTRACK AVEC ESSENTIO OU LIBROSA MAIS PAS LES DEUX
#     onsets_backtrack=np.array([])
#     if len(onsets_complex)>0:
#         onset_frames = time_to_frames(onsets_complex,sr=sampling_rate,hop_length=512)

#         onsets_backtrack = onset_backtrack(onset_frames,pool['odf.complex'])
#         onsets_backtrack = frames_to_time(onsets_backtrack,sr=sampling_rate,hop_length=512)
    
#     return onsets_complex, onsets_backtrack

def find_elems_in_range(array, lower, upper):
    elems=[]
    for elem in array:
        if lower<elem<upper:
            elems.append(elem)
    return elems