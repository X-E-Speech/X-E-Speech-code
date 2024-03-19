from glob import glob
from tqdm import tqdm
import librosa
import soundfile as sf
import os

data_dir = "./dataset"

wav_files = glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
wav_files=sorted(wav_files)

for wavPath in tqdm(wav_files):
    audio, sr = librosa.load(wavPath,sr=None)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sf.write(wavPath, audio, 16000)