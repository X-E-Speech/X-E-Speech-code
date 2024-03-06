import os
import matplotlib.pyplot as plt

from scipy.io.wavfile import write

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import librosa
import commons
import utils
from mel_processing import mel_spectrogram_torch
from data_utils_whisper_hier_multi_pure import adjust_tensor_size
from models_whisper_hier_multi_pure import SynthesizerTrn_3
from utils import load_filepaths_and_text
from text_cn import cleaned_text_to_sequence
from text_cn.symbols import symbols
from text_cn.cleaners import chinese_cleaners1,english_cleaners2

import numpy as np


def get_text_en(text):
    print(text)
    #text=english_cleaners2(text)
    #print(text)
    text_norm = cleaned_text_to_sequence(text)
    #print(text_norm)
    text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps=utils.get_hparams_from_file("./logs/cross-lingual-TTS-en/config.json")

print('len(symbols)',len(symbols))
net_g = SynthesizerTrn_3(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()

_ = net_g.eval()

_ = utils.load_checkpoint("./logs/cross-lingual-TTS-en/G_930000.pth", net_g, None)


def tts_en(text_str,ref_wav_path,count,tmp):
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

    
    title = text_str.replace(' ','') + os.path.basename(ref_wav_path)

    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        #sid = torch.LongTensor([4]).cuda()
        ref_mel= ref_mel.cuda().unsqueeze(0)
        import time
        for i in range(1):
            start=time.time()
            with torch.no_grad():
                audio, attn, mask, *_ = net_g.infer(x_tst, x_tst_lengths, mel=ref_mel,lang=torch.LongTensor([1]).cuda(),max_len=1000, noise_scale=0.667, noise_scale_w=0.8, length_scale=1)#[0][0,0].data.cpu().float().numpy()
            y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length
            y_hat_vocoder,  mask_vocoder = net_g.vocoder(mel.unsqueeze(0).cuda(), torch.LongTensor([mel.size(0)]).cuda(), mel=ref_mel,max_len=1000)#.module.infer(x, x_lengths, mel=ref_mel,lang=lang_id,max_len=1000)

    
    audio = audio[0,0].data.cpu().float().numpy()
    import numpy as np

    tgt=ref_wav_path
    wav_tgt, sr = librosa.load(tgt, sr=hps.data.sampling_rate)
    write(os.path.join("/objexpsum/pro_obj_exp_new/xtts","{}.wav".format(os.path.basename(ref_wav_path))), hps.data.sampling_rate, wav_tgt)
    write(os.path.join("/objexpsum/pro_obj_exp_new/xtts","{}_{}.wav".format(count,os.path.basename(ref_wav_path))), hps.data.sampling_rate, audio)



count=0
from tqdm import tqdm
phone_text=[
    "wˌɛn ɐ mˈæn lˈʊks fɔːɹ sˈʌmθɪŋ bɪjˌɑːnd hɪz ɹˈiːtʃ, hɪz fɹˈɛndz sˈeɪ hiː ɪz lˈʊkɪŋ fɚðə pˈɑːt ʌv ɡˈoʊld æt ðɪ ˈɛnd ʌvðə ɹˈeɪnboʊ.",
    "ɪf ðə ɹˈɛd ʌvðə sˈɛkənd bˈoʊ fˈɔːlz əpˌɑːn ðə ɡɹˈiːn ʌvðə fˈɜːst, ðə ɹɪzˈʌlt ɪz tə ɡˈɪv ɐ bˈoʊ wɪð ɐn ɐbnˈoːɹməli wˈaɪd jˈɛloʊ bˈænd, sˈɪns ɹˈɛd ænd ɡɹˈiːn lˈaɪt wɛn mˈɪkst fˈɔːɹm jˈɛloʊ.",
    "wˌɛn ðə sˈʌnlaɪt stɹˈaɪks ɹˈeɪndɹɑːps ɪnðɪ ˈɛɹ, ðeɪ ˈækt æz ɐ pɹˈɪzəm ænd fˈɔːɹm ɐ ɹˈeɪnboʊ.",
    "wˈʌn sˈiːzən, ðeɪ mˌaɪt dˈuː wˈɛl.",
    "ɪt ɪz dˈeɪndʒɚɹəs ænd ɪt ɪz ɐ lˈaɪ."
]
spk_list=[
    "/jvs_ver1/jvs_preprocessed/jvs_wav_preprocessed/jvs090/VOICEACTRESS100_020.wav"
]
for text in tqdm(phone_text):
    tmp=0    
    for spk in (spk_list):
        tts_en(text,spk,count,tmp)
        tmp=tmp+1
    count=count+1

