import os
import random

import librosa
import matplotlib.pyplot as plt
import torch
from scipy.io.wavfile import write
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from mel_processing import mel_spectrogram_torch


def adjust_tensor_size(tensor, target_size):
    _, t = tensor.size()
    #print(t,target_size)
    while t < target_size:
        tensor=torch.cat((tensor, tensor), dim=1)
        _, t = tensor.size()

    start_idx = random.randint(0, t - target_size)
    adjusted_tensor = tensor[:, start_idx:start_idx + target_size]
    _, t = adjusted_tensor.size()
    #print(t,target_size)
    return adjusted_tensor

import numpy as np

from models_whisper_hier_multi_pure import SynthesizerTrn
from text_cn import cleaned_text_to_sequence
from text_cn.cleaners import chinese_cleaners1, english_cleaners2
from text_cn.symbols import symbols


def get_text_en(text):
    print(text)
    # text=english_cleaners2(text)
    # print(text)
    text_norm = cleaned_text_to_sequence(text)
    print(text_norm)
    text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps=utils.get_hparams_from_file("/logs/cross-lingual-TTS-en/config.json")

print('len(symbols)',len(symbols))
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()

_ = net_g.eval()

_ = utils.load_checkpoint("/logs/cross-lingual-TTS-en/G_650000.pth", net_g, None)


def tts_en(text_str,ref_wav_path):
    stn_tst = get_text_en(text_str)

    # mel_filename = ref_wav_path.replace(".wav", ".f{}h{}w{}mel.pt".format(hps.data.filter_length, hps.data.hop_length, hps.data.win_length))
    # mel = torch.load(mel_filename)
    # print(mel.size())
    # ref_mel=adjust_tensor_size(mel,300)
    # print(ref_mel.size())

    tgt=ref_wav_path
    wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
    wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
    mel = mel_spectrogram_torch(
        wav_tgt, 
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )
    mel = torch.squeeze(mel, 0)
    ref_mel=adjust_tensor_size(mel, 300)
    
    title = os.path.basename(ref_wav_path) + text_str.replace(' ','_')

    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()

        ref_mel= ref_mel.cuda().unsqueeze(0)
        import time
        for i in range(1):
            start=time.time()
            audio, attn, mask, *_ = net_g.infer(x_tst, x_tst_lengths, mel=ref_mel,lang=torch.LongTensor([1]).cuda(),max_len=1000, noise_scale=0.667, noise_scale_w=0.8, length_scale=1)#[0][0,0].data.cpu().float().numpy()
            y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length
            y_hat_vocoder,  mask_vocoder = net_g.vocoder(mel.unsqueeze(0).cuda(), torch.LongTensor([mel.size(0)]).cuda(), mel=ref_mel,max_len=1000)#.module.infer(x, x_lengths, mel=ref_mel,lang=lang_id,max_len=1000)
            print(time.time()-start)
    
    audio = audio[0,0].data.cpu().float().numpy()
    import numpy as np

    tgt=ref_wav_path
    wav_tgt, sr = librosa.load(tgt, sr=hps.data.sampling_rate)
    write(os.path.join("/objexpsum/pro_obj_exp_new/xetts","{}.wav".format(os.path.basename(ref_wav_path))), hps.data.sampling_rate, wav_tgt)

    write(os.path.join('/objexpsum/pro_obj_exp_new/xetts', "{}_{}.wav".format(count,os.path.basename(ref_wav_path))), hps.data.sampling_rate, audio)

count=0
from tqdm import tqdm

phone_text=[
    "bɪkˈʌz hiː wʌzɐ mˈæn wɪð ˈɪnfɪnət ɹᵻsˈoːɹs ænd səɡˈæsᵻɾi.",
    "ðɪs wʌzðə twˈɛnti fˈɪfθ ʌv noʊvˈɛmbɚ, sˈɛvəntˌiːn ˈeɪɾi θɹˈiː.",
    "wˌaɪ ʃˌʊd ˈaɪ kˈɛɹ ðˌoʊ dˈeɪvɪdz lˈɪps wɜː twˈɪtʃɪŋ?",
    "ɪnwˌɪtʃ fˈɑːks lˈuːzᵻz ɐ tˈeɪl ænd ɪts ˈɛldɚ sˈɪstɚ fˈaɪndz wˌʌn.",
    "ˈɔːl smˈaɪl wɜː ɹˈiːəl ænd ðə hˈæpɪɚ ðə mˈoːɹ sɪnsˈɪɹ ."
]
spk_list=[
    "/0008/Angry/0008_000556.wav",
    "/0003/Sad/0003_001388.wav",
    "/0005/Happy/0005_000708.wav",
    "/0010/Sad/0010_001054.wav",
    "/0007/Happy/0007_000707.wav",
    "/0002/Angry/0002_000365.wav"
]
for text in tqdm(phone_text):
    tmp=0    
    for spk in (spk_list):
        tts_en(text,spk)
        tmp=tmp+1
    count=count+1

