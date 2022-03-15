import numpy as np
import librosa as lr
import h5py
import time
import os
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import matplotlib.pyplot as plt
import IPython.display as ipd

from skimage.util.shape import view_as_windows

PATH = '/mnt/c/svdcvt/teacher/sirius/'
PATH1 = os.path.join(PATH, 'dataset/')


def to_melspec(ar, n_fft=2048):
    return lr.amplitude_to_db(lr.feature.melspectrogram(ar, n_fft=n_fft))

def file_to_segments(file, sr=22050, duration=3, step=2):
    if isinstance(file, (str)):
        file, sr = lr.load(path=file, sr=sr)
    elif isinstance(file, (np.ndarray)):
        pass
    else:
        raise ValueError()
    
    file_duration = len(file) / sr
    segm_ticks = duration * sr
    step_ticks = step * sr
    if file_duration < duration:
        segments = np.tile(file, segm_ticks // len(file) + 1)[:segm_ticks][None, :]
    else:
        segments = view_as_windows(file, window_shape=(segm_ticks, ), step=step_ticks)
    del file
    return segments
    
def track_to_spectrograms(track_path, duration=3, sr=22050, n_fft=2048, save_path='./dataset.hdf5'):
    '''
    Arguments:
    path: str 
        list of paths to audio samples
    duration: float
        duration of segment in seconds
    sr: int
        sample rate
    n_fft: int 
        parameter of stft (other parameters need to be default)
    save_path: str
        saves dataset to .hdf5 format
        
    '''
    segments = file_to_segments(track_path, sr=sr, duration=duration, step=duration)
    melspec_segments = []
    for i, segment in enumerate(segments):
        melspec_segments.append(to_melspec(segment))
    return melspec_segments