import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import torch
import random
from tqdm import tqdm
from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram


def load_model(path) -> Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location="cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    print(device,dims)
    model = Whisper(dims)
    del model.decoder
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    model.half()
    model.to(device)
    return model


def pred_ppg(whisper: Whisper, wavPath, ppgPath):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppgln = audln // 320
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).half().to(whisper.device)
    with torch.no_grad():
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
        ppg = ppg[:ppgln,] # [length, dim=1024]
        # print(ppg.shape)
        os.makedirs(os.path.dirname(ppgPath), exist_ok=True)  # Create the directory if it doesn't exist
        np.save(ppgPath, ppg, allow_pickle=False)

def pred_ppg_infer(whisper: Whisper, wavPath, ppgPath):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppgln = audln // 320
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).half().to(whisper.device)
    with torch.no_grad():
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
        ppg = ppg[:ppgln,] # [length, dim=1024]
    return ppg
        # # print(ppg.shape)
        # os.makedirs(os.path.dirname(ppgPath), exist_ok=True)  # Create the directory if it doesn't exist
        # np.save(ppgPath, ppg, allow_pickle=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-p", "--ppg", help="ppg", dest="ppg")
    args = parser.parse_args()
    print(args.wav)
    print(args.ppg)

    os.makedirs(args.ppg, exist_ok=True)
    wavPath = args.wav
    ppgPath = args.ppg

    whisper = load_model('./whisper_pretrain/large-v2.pt')

    for root, dirs, files in os.walk(wavPath):
        for file in tqdm(files, desc='Processing WAV files'):
            if file.endswith(".wav"):
                relative_path = os.path.relpath(os.path.join(root, file), wavPath)
                path_wav = os.path.join(wavPath, relative_path)
                path_ppg = os.path.join(ppgPath, os.path.splitext(relative_path)[0] + "_largev2ppg")
                if os.path.isfile(f"{path_ppg}.npy"):
                    continue
                pred_ppg(whisper, path_wav, path_ppg)
