# X-E-Speech: Joint Training Framework of Non-Autoregressive Cross-lingual Emotional Text-to-Speech and Voice Conversion


[![Openreview](https://img.shields.io/badge/Anonymous-preprint-<COLOR>.svg)](https://openreview.net/forum?id=J4fL6FDz36)
[![githubio](https://img.shields.io/static/v1?message=Audio%20Samples&logo=Github&labelColor=grey&color=blue&logoColor=white&label=%20&style=flat)](https://X-E-Speech.github.io/X-E-Speech-demopage)
![GitHub Repo stars](https://img.shields.io/github/stars/X-E-Speech/X-E-Speech-code)
![GitHub](https://img.shields.io/github/license/X-E-Speech/X-E-Speech-code)



In this [paper](https://openreview.net/forum?id=J4fL6FDz36), we propose a cross-lingual emotional speech generation model, X-E-Speech, which achieves the disentanglement of speaker style and cross-lingual content features by jointly training non-autoregressive (NAR) voice conversion (VC) and text-to-speech (TTS) models. For TTS, we freeze the style-related model components and fine-tune the content-related structures to enable cross-lingual emotional speech synthesis without accent. For VC, we improve the emotion similarity between the generated results and the reference speech by introducing the similarity loss between content features for VC and text for TTS.

Visit our [demo page](https://X-E-Speech.github.io/X-E-Speech-demopage) for audio samples.

We also provide the [pretrained models](https://drive.google.com/drive/folders/1PHzFyqkOa_7O4TVI6vypZa8MIpU7nIbT?usp=sharing).

<table style="width:100%">
  <tr>
    <td><img src="x-speech-biger.png"  height="300"></td>
  </tr>
</table>

## Todo list

- Better inference code and instructions for inference code
- [HuggingFace](https://huggingface.co/x-e-speech/x-e-speech) or Colab

## Pre-requisites

1. Clone this repo: `git clone https://github.com/X-E-Speech/X-E-Speech-code.git`

2. CD into this repo: `cd X-E-Speech-code`

3. Python=3.7, Install python requirements: `pip install -r requirements.txt`
   
   You may need to install:
   1. espeak for English: `apt-get install espeak`
   2. pypinyin `pip install pypinyin` and [jieba](https://github.com/fxsjy/jieba) for Chinese
   3. pyopenjtalk for Japenese: `pip install pyopenjtalk`

4. Download [Whisper-large-v2](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt) and put it under directory 'whisper-pretrain/'

5. Download the [VCTK](https://datashare.ed.ac.uk/handle/10283/3443), [Aishell3](https://www.openslr.org/93/), [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) dataset (for training cross-lingual TTS and VC)

6. Download the [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) dataset (for training cross-lingual emotional TTS and VC)

7. Build Monotonic Alignment Search
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace
```

## Inference Example

Download the pretrained checkpoints and run:

```python
#For Cross-lingual Chinese TTS
inference-cross-lingual-TTS-cn.py
#For Cross-lingual English TTS
inference-cross-lingual-TTS-en.py
#For Cross-lingual Emotional English TTS
inference-cross-lingual-emotional-TTS-en.py
#For Cross-lingual Emotional VC, you need to refer to preprocess_weo.py to generate npy files first
inference-cross-lingual-emotional-VC.py

```

## Training Example

1. Preprocess-resample to 16KHz

Copy datasets to the `dataset` folder and then resample the audios to 16KHz by `dataset/downsample.py`.
This will rewrite the original wav files, so please copy but not cut your original dataset!


2. Preprocess-whisper

Generate the whisper encoder output.

```python
python preprocess_weo.py  -w dataset/vctk/ -p dataset/vctk_largev2
python preprocess_weo.py  -w dataset/aishell3/ -p dataset/aishell3_largev2
python preprocess_weo.py  -w dataset/jvs/ -p dataset/jvs_largev2
python preprocess_weo.py  -w dataset/ESD/ -p dataset/ESD_largev2
```

3. Preprocess-g2p

I provide the g2p results for my dataset in `filelist`. If you want to do g2p to your datasets:

- For English, refer to `preprocess_en.py` and [VITS](https://github.com/jaywalnut310/vits/blob/main/preprocess.py);
- For Japanese, refer to `preprocess_jvs.py` and [VITS-jvs](https://github.com/zassou65535/VITS);
- For Chinese, use `jieba_all.py` to split the words and then use the `preprocess_cn.py` to generate the pinyin.

Refer to `filelist/train_test_split.py` to split the dataset into train set and test set.

4. Train cross-lingual TTS and VC

Train the whole model by cross-lingual datasets:

```python
python train_whisper_hier_multi_pure_3.py  -c configs/cross-lingual.json -m cross-lingual-TTS
```

Freeze the speaker-related part and finetune the content related part by mono-lingual dataset:

```python
python train_whisper_hier_multi_pure_3_freeze.py  -c configs/cross-lingual-emotional-freezefinetune-en.json -m cross-lingual-TTS-en
```

5. Train cross-lingual emotional TTS and VC

Train the whole model by cross-lingual emotional datasets:

```python
python train_whisper_hier_multi_pure_esd.py  -c configs/cross-lingual-emotional.json -m cross-lingual-emotional-TTS
```

Freeze the speaker-related part and finetune the content related part by mono-lingual dataset:

```python
python train_whisper_hier_multi_pure_esd_freeze.py  -c configs/cross-lingual-emotional-freezefinetune-en.json -m cross-lingual-emotional-TTS-en
```

## Change the model structure

There is two SynthesizerTrn in models_whisper_hier_multi_pure.py. The difference is the [n_langs](https://github.com/X-E-Speech/X-E-Speech-code/blob/9d2a3b0132a066f26aaa16fce3b126542688401a/models_whisper_hier_multi_pure.py#L869).

So if you want to train this model for more than 3 languages, change the number of n_langs.


## References

- https://github.com/jaywalnut310/vits
- https://github.com/PlayVoice/lora-svc
- https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion
- https://github.com/OlaWod/FreeVC/blob/main/README.md
- https://github.com/zassou65535/VITS

---

- Have questions or need assistance? Feel free to [open an issue](https://github.com/X-E-Speech/X-E-Speech-code/issues/new), I will try my best to help you. 
- Welcome to give your advice to me!
- If you like this work, please give me a star on GitHub! ⭐️ It encourages me a lot.
