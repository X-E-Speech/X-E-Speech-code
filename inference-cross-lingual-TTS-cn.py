import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils_whisper_hier_multi_pure import adjust_tensor_size
from scipy.io.wavfile import write
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from mel_processing import mel_spectrogram_torch
from models_whisper_hier_multi_pure import SynthesizerTrn, SynthesizerTrn_3
from text_cn import cleaned_text_to_sequence
from text_cn.cleaners import chinese_cleaners1, english_cleaners2
from text_cn.symbols import symbols


def get_text_cn(text):
    # print(text)
    # text=chinese_cleaners1(text)
    print(text)
    text_norm = cleaned_text_to_sequence(text)
    print(text_norm)
    text_norm = torch.LongTensor(text_norm)
    return text_norm



hps=utils.get_hparams_from_file("/logs/cross-lingual-TTS-cn/config.json")

print('len(symbols)',len(symbols))
net_g = SynthesizerTrn_3(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()

_ = net_g.eval()

_ = utils.load_checkpoint("/logs/cross-lingual-TTS-cn/G_830000.pth", net_g, None)


def tts_en(text_str,ref_wav_path,count,tmp):
    stn_tst = get_text_cn(text_str)

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
                audio, attn, mask, *_ = net_g.infer(x_tst, x_tst_lengths, mel=ref_mel,lang=torch.LongTensor([0]).cuda(),max_len=1000, noise_scale=0.667, noise_scale_w=0.8, length_scale=1)#[0][0,0].data.cpu().float().numpy()
            y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length
            y_hat_vocoder,  mask_vocoder = net_g.vocoder(mel.unsqueeze(0).cuda(), torch.LongTensor([mel.size(0)]).cuda(), mel=ref_mel,max_len=1000)#.module.infer(x, x_lengths, mel=ref_mel,lang=lang_id,max_len=1000)
            #print(time.time()-start)
    
    audio = audio[0,0].data.cpu().float().numpy()
    import numpy as np

    tgt=ref_wav_path
    wav_tgt, sr = librosa.load(tgt, sr=hps.data.sampling_rate)
    write(os.path.join("/objexpsum/chinese/pro","{}.wav".format(os.path.basename(ref_wav_path))), hps.data.sampling_rate, wav_tgt)
    write(os.path.join("/objexpsum/chinese/pro","{}_{}.wav".format(count,os.path.basename(ref_wav_path))), hps.data.sampling_rate, audio)


count=0
from tqdm import tqdm

phone_text=[
    "ji4 - liu4 yue4 - chong1 - gao1 - hou4 - you3 suo3 - hui2 luo4",
    "xiao4 hua1 de5 tie1 shen1 gao1 shou3 - shu1 ji2 - you3 - shen2 me5",
    "cong2 lai2 - mei2 - zuo4 - guo4 - zhe4 mo2 - ke3 pa4 - de5 - meng4",
    "yi1 yuan4 - men2 kou3 - de5 - zhao1 pai2 - shang4 - xie3 - zhe5",
    "zai4 - dui4 dai4 - zhong1 gai4 gu3 - hui2 gui1 - de5 - wen4 ti2 - shang4",
    "hong2 da2 xin1 cai2 - fa1 bu4 - chong2 zu3 - cao3 an4",
    "lu2 ka3 si1 - miao2 shu4 - le5 - lu2 ka3 si1 - de5 - suo3 jian4 suo3 wen2",
    "er2 qie3 - zhun3 bei4 - hao3 - lai2 - dui4 fu4 - ta1 men5",
    "guang3 qu2 - lu4 kou3 - ding1 zi4 - lu4 kou3",
    "bai2 yang2 zuo4 - de5 - zhu3 chi2 ren2 - you3 - shen2 me5"
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