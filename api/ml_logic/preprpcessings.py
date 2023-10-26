import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import base64
from io import BytesIO

def load_audio(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    # y_denoised = nr.reduce_noise(y=y, sr=sr)
    y_trimmed, index = librosa.effects.trim(y, top_db=20)
    return y_trimmed


def get_spectrogram(y):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max),
        y_axis="log",
        x_axis="time"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    return image


def extract_mfcc(
        audio_file_path,
        sample_rate=None,
        pre_emphasis=0.97,
        frame_size=0.025,
        frame_stride=0.01,
        NFFT=512,
        nfilt=40,
        num_ceps=12,
        fixed_length=100
):
    """
    Extracts Mel-frequency cepstral coefficients (MFCC) from an audio file.

    Parameters:
    - audio_file_path (str): Path to the input audio file.
    - sample_rate (int, optional): Desired sample rate for the audio. If None, uses the original audio's sample rate.
    - pre_emphasis (float, optional): Pre-emphasis filter coefficient. Default is 0.97.
    - frame_size (float, optional): Frame size in seconds. Default is 0.025.
    - frame_stride (float, optional): Frame stride in seconds. Default is 0.01.
    - NFFT (int, optional): Number of FFT bins. Default is 512.
    - nfilt (int, optional): Number of Mel filters. Default is 40.
    - num_ceps (int, optional): Number of cepstral coefficients to return. Default is 12.
    - fixed_length (int, optional): The fixed number of MFCC frames to return. Default is 100.

    Returns:
    - np.ndarray: An array of shape (fixed_length, num_ceps) containing the extracted MFCCs.

    """

    # Load the audio file using librosa
    y, sr = librosa.load(audio_file_path, sr=sample_rate)

    # Apply the pre-emphasis filter
    emphasized_signal = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # Calculate frame length and step in samples
    frame_length, frame_step = frame_size * sr, frame_stride * sr
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    # Calculate the number of frames
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    # Pad the signal so that all frames have equal number of samples without truncating any samples from the original signal
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    # Create frames using stride tricks
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Apply window function (Hamming window)
    frames *= np.hamming(frame_length)

    # Compute power spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # Convert Hz to Mel scale
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sr)

    # Create filter banks
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    # Filter the power spectrum through the filter banks
    filter_banks = np.dot(pow_frames, fbank.T)

    # Avoid numerical issues by replacing zero values
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)

    # Convert to dB
    filter_banks = 20 * np.log10(filter_banks)

    # Compute MFCC using DCT
    from scipy.fftpack import dct
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]

    # Adjust the length of MFCC sequence to a fixed length
    if mfcc.shape[0] < fixed_length:
        pad_width = fixed_length - mfcc.shape[0]
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:fixed_length]

    return mfcc


def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


