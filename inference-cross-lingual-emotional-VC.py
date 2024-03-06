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
from models_whisper_hier_multi_pure import SynthesizerTrn

from text_cn import cleaned_text_to_sequence
from text_cn.symbols import symbols
from text_cn.cleaners import chinese_cleaners1,english_cleaners2

import numpy as np




hps=utils.get_hparams_from_file("/logs/cross-lingual-emotional-VC/config.json")

print('len(symbols)',len(symbols))
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()

_ = net_g.eval()

_ = utils.load_checkpoint("/logs/cross-lingual-emotional-VC/G_450000.pth", net_g, None)


def tts_en(text_str,ref_wav_path):
    src_wav=text_str
    weo_filename = src_wav.replace(".wav", "_largev2ppg.npy")
    weo_filename = weo_filename.replace("/ESD_16k/","/ESD_16k_largev2/")
    weo =torch.from_numpy(np.load(weo_filename))
    weo=weo.transpose(1,0)

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
    
    title = "to"+os.path.basename(ref_wav_path) +"from"+ os.path.basename(src_wav)

    with torch.no_grad():
        x_tst = weo.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([weo.size(0)]).cuda()
        #sid = torch.LongTensor([4]).cuda()
        ref_mel= ref_mel.cuda().unsqueeze(0)
        import time
        for i in range(1):
            start=time.time()
            audio, *_ = net_g.voice_conversion_new(x_tst, x_tst_lengths, mel=ref_mel,lang=torch.LongTensor([1]).cuda(),max_len=1000)#[0][0,0].data.cpu().float().numpy()
            # y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length
            # y_hat_vocoder,  mask_vocoder = net_g.vocoder(mel.unsqueeze(0).cuda(), torch.LongTensor([mel.size(0)]).cuda(), mel=ref_mel,max_len=1000)#.module.infer(x, x_lengths, mel=ref_mel,lang=lang_id,max_len=1000)
            # print(time.time()-start)
    
    audio = audio[0,0].data.cpu().float().numpy()

    # print(np.max(audio),np.min(audio),audio.shape,audio.dtype)
    # write("audio.wav", 16000, audio)
    tgt=ref_wav_path#'/home/dataset/ESD/ESD/ENG22050/0014/Sad/0014_001057.wav'
    wav_tgt, sr = librosa.load(tgt, sr=hps.data.sampling_rate)
    write(os.path.join("/objexpsum/pro_obj_exp_new/xevc","{}.wav".format(os.path.basename(ref_wav_path))), hps.data.sampling_rate, wav_tgt)
    
    wav_src, sr = librosa.load(src_wav, sr=hps.data.sampling_rate)
    write(os.path.join("/objexpsum/pro_obj_exp_new/xevc","{}.wav".format(os.path.basename(src_wav))), hps.data.sampling_rate, wav_src)

    write(os.path.join('/objexpsum/pro_obj_exp_new/xevc', f"{title}.wav"), hps.data.sampling_rate, audio)

count=0
from tqdm import tqdm
phone_text=[
    "/0019/Angry/0019_000359.wav",
    "/0017/Surprise/0017_001407.wav",
    "/0015/Angry/0015_000358.wav",
    "/0014/Sad/0014_001057.wav",
    "/0013/Sad/0013_001062.wav",
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

