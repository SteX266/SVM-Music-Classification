
import os
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np






def split_files(genres):
    i=0
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


def make_spectrogram(genres):
    for g in genres:
        j = 0
        print(g)
        for filename in os.listdir(os.path.join('./Data/genres_original_3sec', f"{g}")):
            audio_path = f'./Data/genres_original_3sec/{g}/' + filename
            audio, sr = librosa.load(audio_path)
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)






if __name__ == '__main__':
    genres = 'blues classical country disco pop hiphop metal reggae rock jazz'
    genres = genres.split()
 #   split_files(genres)
    make_spectrogram(genres)