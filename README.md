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
    <td><img src="x-speech-biger.png"  height="400"></td>
  </tr>
</table>

## Todo list

- Better inference code.
- Better readme
- HuggingFace or Colab


The generation of whisper encoder output refer to: https://github.com/PlayVoice/lora-svc
(I will upload the code and instruction for data process later)

The inference and train codes are available now, the environment is similar to VITS.

The pre-trained models are available here: https://drive.google.com/drive/folders/1PHzFyqkOa_7O4TVI6vypZa8MIpU7nIbT?usp=sharing

I will write a better readme and manage the code in the following days. 


## References

- https://github.com/jaywalnut310/vits
- https://github.com/PlayVoice/lora-svc
- https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion
