import os
import librosa.feature
import numpy as np
import pandas as pd

# Initializing an empty list to store extracted features
features = []

# Specifying the directory containing audio files
audio_folder = "../data/input_data/audio/tests"

# Defining the names of the features to be extracted
feature_names = [
    "Audio_File",
    "Amplitude_Envelope_Mean",
    "RMS_Mean",
    "ZCR_Mean",
    "STFT_Mean",
    "SC_Mean",
    "SBAN_Mean",
    "SCON_Mean",
    "MelSpec",
    *["MFCCs" + str(i) for i in range(1, 14)],
    "Cry_Reason"
]


# Function to extract features from an audio file
def extract_features(file):
    # Loading the audio file
    audio_signal, sampling_rate = librosa.load(file)

    # Calculating the mean amplitude envelope
    amp_env = np.mean(np.abs(audio_signal))
    # Calculating the mean RMS (Root Mean Square) value
    rms = np.mean(librosa.feature.rms(y=audio_signal))
    # Calculating the mean Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio_signal))
    # Calculating the mean of the STFT (Short Time Fourier Transform)
    stft = np.abs(librosa.stft(audio_signal))
    stft_mean = np.mean(stft)
    # Calculating the mean spectral centroid
    sc = np.mean(librosa.feature.spectral_centroid(y=audio_signal, sr=sampling_rate))
    # Calculating the mean spectral contrast
    scon = np.mean(librosa.feature.spectral_contrast(y=audio_signal, sr=sampling_rate))
    # Calculating the mean spectral bandwidth
    sban = np.mean(librosa.feature.spectral_bandwidth(y=audio_signal, sr=sampling_rate))
    # Calculating the mean of the mel spectrogram
    melspec = np.mean(librosa.feature.melspectrogram(y=audio_signal, sr=sampling_rate))
    # Calculating the mean of the first 13 MFCCs (Mel Frequency Cepstral Coefficients)
    mfccs = np.mean(librosa.feature.mfcc(y=audio_signal, sr=sampling_rate, n_mfcc=13), axis=1)

    # Returning a list of extracted features
    return [file, amp_env, rms, zcr, stft_mean, sc, sban, scon, melspec, *mfccs]


# Iterating over each folder in the specified directory
for folder in os.listdir(audio_folder):
    folder_path = os.path.join(audio_folder, folder)

    # Checking if the path is a directory
    if os.path.isdir(folder_path):
        # Iterating over each audio file in the folder
        for audio_file in os.listdir(folder_path):
            # Checking if the file extension is '.wav'
            if audio_file.endswith(".wav"):
                file_path = os.path.join(folder_path, audio_file)
                # Extracting features and appending to the 'features' list along with the reason of cry (name of the subfolder)
                features.append(extract_features(file_path) + [folder])

# Creating a DataFrame to hold all the extracted features with their respective column names
features_df = pd.DataFrame(features, columns=feature_names)

# Saving the extracted features to a CSV file
features_df.to_csv("../data/input_data/transformed/test_features_2.csv", index=False)
