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
    <td><img src="x-speech-biger.png"  height="350"></td>
  </tr>
</table>

## Todo list

- Better inference code.
- Better readme
- HuggingFace or Colab

## Pre-requisites

1. Clone this repo: `git clone https://github.com/X-E-Speech/X-E-Speech-code.git`

2. CD into this repo: `cd X-E-Speech-code`

3. Install python requirements: `pip install -r requirements.txt`

4. Download [Whisper-large-v2](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt) and put it under directory 'whisper-pretrain/'

5. Download the [VCTK](https://datashare.ed.ac.uk/handle/10283/3443), [Aishell3](https://www.openslr.org/93/), [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) dataset (for training cross-lingual TTS and VC)

6. Download the [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) dataset (for training cross-lingual emotional TTS and VC)


## Inference Example

Download the pretrained checkpoints and run:

```python

```

## Training Example

1. Preprocess

```python


```

2. Train

```python

```

## References

- https://github.com/jaywalnut310/vits
- https://github.com/PlayVoice/lora-svc
- https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion
- https://github.com/OlaWod/FreeVC/blob/main/README.md
