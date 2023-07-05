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
    features = {"genre": [], "zero_crossing_rate": [], "spectral_centroid": []}
    for g in genres:

        print(g)
        for filename in os.listdir(os.path.join('./Data/genres_original_3sec', f"{g}")):
            audio_path = f'./Data/genres_original_3sec/{g}/' + filename
            audio, sr = librosa.load(audio_path)

            zero_crossing_rate = calculate_zero_crossing_rate(audio)

            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

            spectral_centroid = calculate_spectral_centroid(mel_spectrogram, sr)

            features["genre"].append(g)
            features["zero_crossing_rate"].append(zero_crossing_rate)
            features["spectral_centroid"].append(spectral_centroid)
    save_features_to_csv(features)


def save_features_to_csv(features):
    csv_file = './features.csv'

    # Extract the keys from the dictionary to use as the CSV header
    header = features.keys()

    # Extract the values from the dictionary
    rows = zip(*features.values())

    # Write the dictionary to the CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def calculate_zero_crossing_rate(audio):
    zero_crossings = librosa.zero_crossings(audio, pad=False)
    zero_crossing_rate = np.mean(zero_crossings)
    return zero_crossing_rate


def calculate_spectral_centroid(mel_spectrogram, sr):
    spectral_centroids = librosa.feature.spectral_centroid(S=mel_spectrogram, sr=sr)
    single_centroid = spectral_centroids.mean()
    return single_centroid


if __name__ == '__main__':
    genres = 'blues classical country disco pop hiphop metal reggae rock jazz'
    genres = genres.split()
    #   split_files(genres)
    generate_features_csv(genres)
