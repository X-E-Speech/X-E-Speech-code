import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons 
from mel_processing import spectrogram_torch, mel_spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text

from text_cn import text_to_sequence, cleaned_text_to_sequence
import utils


def adjust_tensor_size(tensor, target_size):
    _, t = tensor.size()
    while t < target_size:
        tensor=torch.cat((tensor, tensor), dim=1)
        _, t = tensor.size()

    start_idx = random.randint(0, t - target_size)
    adjusted_tensor = tensor[:, start_idx:start_idx + target_size]
    _, t = adjusted_tensor.size()
    return adjusted_tensor


""""""
"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.max_wav_value  = hparams.max_wav_value
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length 
        self.hop_length     = hparams.hop_length 
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate 
        self.hps = utils.get_hparams()
        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        self.target_size=300 #16000khz 320hop 300 means 6s
        print("The target size of reference speech",self.target_size)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()


    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):

        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        

        if "/aishell3_train" in audiopath:
            lang_id=0
        elif "/vctk" in audiopath:
            lang_id=1
        elif "chinese(justlikeaishell3)" in audiopath:
            lang_id=0
        elif "english(justlikevctk)" in audiopath:
            lang_id=1
        elif "/biaobei" in audiopath:
            lang_id=0
        elif "/ljspeech" in audiopath:
            lang_id=1
        elif "/jvs_wav_preprocessed" in audiopath:
            lang_id=2
        elif "/jsut_basic5000_16k" in audiopath:
            lang_id=2
        else:
            print(audiopath,"没有lang id")


        text = self.get_text_lang(text,lang_id)
        spec,weo, wav,ref_mel = self.get_audio(audiopath)


        if text.size(0)>spec.size(1):
            print(audiopath)
            print(text.size(),spec.size())

        return (text, spec, weo,wav,ref_mel,lang_id)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(" {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)

        weo_filename = filename.replace(".wav", "_largev2ppg.npy")

        weo_filename = weo_filename.replace("/vctk/","/vctk_largev2/")
        weo_filename = weo_filename.replace("/aishell3_train/","/aishell3_train_largev2/")
        weo_filename = weo_filename.replace("/ESD_16k/","/ESD_16k_largev2/")
        weo_filename = weo_filename.replace("/jvs_wav_preprocessed/","/jvs_wav_preprocessed_largev2/")

        weo =torch.from_numpy(np.load(weo_filename))
        weo=weo.transpose(1,0)

 
        
        mel_filename = filename.replace(".wav", ".f{}h{}w{}mel.pt".format(self.filter_length, self.hop_length, self.win_length))
        if os.path.exists(mel_filename):
            mel = torch.load(mel_filename)
 
        else:

            mel = mel_spectrogram_torch(
                audio_norm, 
                self.hps.data.filter_length, 
                self.hps.data.n_mel_channels, 
                self.hps.data.sampling_rate, 
                self.hps.data.hop_length, 
                self.hps.data.win_length, 
                self.hps.data.mel_fmin, 
                self.hps.data.mel_fmax)

            mel = torch.squeeze(mel, 0)
            torch.save(mel, mel_filename)

        spec=mel

        if spec.size(1)!=weo.size(1):
            print(filename, spec.size(1),weo.size(1),'spec.size()[1]!=weo.size()[1]')
        ref_mel=adjust_tensor_size(mel, self.target_size)
    
        if ref_mel.size(1)!=self.target_size:
            print('mel.size()[1]!=self.target_size')

        return spec,weo, audio_norm,ref_mel


    def get_text(self, text):
        text_norm = []

        text_norm = cleaned_text_to_sequence(text)
        #print(text,text_norm)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
            import sys
            sys.exit()
        
            
        text_norm = torch.LongTensor(text_norm)

        return text_norm
    def get_text_lang(self, text,lang):
        text_norm = []

        text_norm = cleaned_text_to_sequence(text)

        if lang==1:
            text_norm = commons.intersperse(text_norm, 0)
        elif lang!=0 and lang!=2:
            print(lang)
        
 
        text_norm = torch.LongTensor(text_norm)

        return text_norm
    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)
        #(text, spec, weo,wav,ref_mel,lang_id)
        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_weo_len = max([x[2].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])
        max_mel_len=max([x[4].size(1) for x in batch])
        if max_weo_len!=max_spec_len:
            print('max_weo_len!=max_spec_len')
        if max_mel_len!=300:
            print('max_mel_len!=300')


        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        weo_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        lang_id = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        weo_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_weo_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        mel_padded = torch.FloatTensor(len(batch), batch[0][4].size(0), max_mel_len)


        text_padded.zero_()
        spec_padded.zero_()
        weo_padded.zero_()
        wav_padded.zero_()
        mel_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]


            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            weo = row[2]
            weo_padded[i, :, :weo.size(1)] = weo
            weo_lengths[i] = weo.size(1)
            if weo_lengths[i]!=spec_lengths[i]:
                print(weo_lengths[i]!=spec_lengths[i])

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            mel = row[4]

            mel_padded[i, :, :mel.size(1)] = mel

            lang_id[i] = row[5]



        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths,weo_padded, weo_lengths, wav_padded, wav_lengths, mel_padded,lang_id, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths,weo_padded, weo_lengths, wav_padded, wav_lengths, mel_padded,lang_id


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()

        print(self.num_samples_per_bucket)
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
