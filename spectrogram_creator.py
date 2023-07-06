import csv
import os
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np


def split_files(genres):
    i = 0
    for g in genres:
        j = -1
        print(f"{g}")
        for filename in os.listdir(os.path.join('./Data/genres_original', f"{g}")):

            song = os.path.join(f'./Data/genres_original/{g}', f'{filename}')
            j = j + 1
            for w in range(0, 10):
                i = i + 1
                # print(i)
                t1 = 3 * (w) * 1000
                t2 = 3 * (w + 1) * 1000
                newAudio = AudioSegment.from_wav(song)
                new = newAudio[t1:t2]
                new.export(f'./Data/genres_original_3sec/{g}/{g + str(j) + str(w)}.wav', format="wav")


def generate_features_csv(genres):
    features = {"genre": [], "zero_crossing_rate": [], "spectral_centroid": [],
                "mfcc": [], "spectral_contrast": [], "spectral_rolloff": [], "spectral_bandwidth": []}

    for g in genres:
        print(g)
        for filename in os.listdir(os.path.join('./Data/genres_original_3sec', f"{g}")):
            audio_path = f'./Data/genres_original_3sec/{g}/' + filename
            audio, sr = librosa.load(audio_path)

            zero_crossing_rate = calculate_zero_crossing_rate(audio)
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
            spectral_centroid = calculate_spectral_centroid(mel_spectrogram, sr)
            mfcc = calculate_mfcc(mel_spectrogram, sr)
            spectral_contrast = calculate_spectral_contrast(mel_spectrogram, sr)
            spectral_rolloff = calculate_spectral_rolloff(mel_spectrogram, sr)
            spectral_bandwidth = calculate_spectral_bandwidth(mel_spectrogram, sr)

            features["genre"].append(g)
            features["zero_crossing_rate"].append(zero_crossing_rate)
            features["spectral_centroid"].append(spectral_centroid)
            features["mfcc"].append(mfcc)
            features["spectral_contrast"].append(spectral_contrast)
            features["spectral_rolloff"].append(spectral_rolloff)
            features["spectral_bandwidth"].append(spectral_bandwidth)

    save_features_to_csv(features)


# Modify the save_features_to_csv function to include the new feature headers
def save_features_to_csv(features):
    csv_file = './features.csv'
    header = features.keys()
    rows = zip(*features.values())
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# Modify the existing calculate_zero_crossing_rate function
def calculate_zero_crossing_rate(audio):
    zero_crossings = librosa.zero_crossings(audio, pad=False)
    zero_crossing_rate = np.mean(zero_crossings)
    return zero_crossing_rate


# Modify the existing calculate_spectral_centroid function
def calculate_spectral_centroid(mel_spectrogram, sr):
    spectral_centroids = librosa.feature.spectral_centroid(S=mel_spectrogram, sr=sr)
    single_centroid = spectral_centroids.mean()
    return single_centroid


# New function to calculate MFCCs
def calculate_mfcc(mel_spectrogram, sr):
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), sr=sr)
    mean_mfcc = np.mean(mfcc)
    return mean_mfcc


# New function to calculate spectral contrast
def calculate_spectral_contrast(mel_spectrogram, sr):
    contrast = librosa.feature.spectral_contrast(S=mel_spectrogram, sr=sr)
    mean_contrast = np.mean(contrast)
    return mean_contrast


# New function to calculate spectral rolloff
def calculate_spectral_rolloff(mel_spectrogram, sr):
    rolloff = librosa.feature.spectral_rolloff(S=mel_spectrogram, sr=sr)
    single_rolloff = np.mean(rolloff)
    return single_rolloff


# New function to calculate spectral bandwidth
def calculate_spectral_bandwidth(mel_spectrogram, sr):
    bandwidth = librosa.feature.spectral_bandwidth(S=mel_spectrogram, sr=sr)
    single_bandwidth = np.mean(bandwidth)
    return single_bandwidth
if __name__ == '__main__':
    genres = 'blues classical country disco pop hiphop metal reggae rock jazz'
    genres = genres.split()
    #   split_files(genres)
    generate_features_csv(genres)
