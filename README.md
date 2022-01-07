# Meta-StyleSpeech : Multi-Speaker Adaptive Text-to-Speech Generation


### Eval results (eval on VCTK)

* source: stylespeech; target: meta-stylespeech 
```bash
ifgsm  [ref = 00_gt, test_dir = 06_synthesized_black] asv = 0.6033 +- 0.0757, mos = 3.2930 += 0.1715
mifgsm [ref = 00_gt, test_dir = 06_synthesized_black] asv = 0.6153 +- 0.0635, mos = 3.3004 += 0.1725
pgd    [ref = 00_gt, test_dir = 06_synthesized_black] asv = 0.6098 +- 0.0642, mos = 3.3023 += 0.1696
```
-----------------------------------------------
* source: meta-stylespeech; target: stylespeech (PGD)
```bash
[ref = 00_gt, test_dir =      01_ori_with_adv] asv = 0.8423 +- 0.0681, mos = 3.5113 += 0.0829
[ref = 00_gt, test_dir =     02_ori_with_base] asv = 0.8511 +- 0.0657, mos = 3.4609 += 0.0967
[ref = 00_gt, test_dir =   03_synthesized_ori] asv = 0.7087 +- 0.0815, mos = 3.3328 += 0.1824
[ref = 00_gt, test_dir =   04_synthesized_adv] asv = 0.5622 +- 0.1028, mos = 3.1224 += 0.1918
[ref = 00_gt, test_dir =  05_synthesized_base] asv = 0.6624 +- 0.1075, mos = 3.2340 += 0.2166
[ref = 00_gt, test_dir = 06_synthesized_black] asv = 0.5692 +- 0.0978, mos = 3.1085 += 0.1692
```
* source: meta-stylespeech; target: stylespeech (mifgsm)
```bash
[ref = 00_gt, test_dir =      01_ori_with_adv] asv = 0.8390 +- 0.0692, mos = 3.4886 += 0.0883
[ref = 00_gt, test_dir =     02_ori_with_base] asv = 0.8511 +- 0.0655, mos = 3.4609 += 0.0946
[ref = 00_gt, test_dir =   03_synthesized_ori] asv = 0.7087 +- 0.0815, mos = 3.3328 += 0.1824
[ref = 00_gt, test_dir =   04_synthesized_adv] asv = 0.5921 +- 0.0994, mos = 3.2047 += 0.2021
[ref = 00_gt, test_dir =  05_synthesized_base] asv = 0.6627 +- 0.1047, mos = 3.2310 += 0.2258
[ref = 00_gt, test_dir = 06_synthesized_black] asv = 0.5807 +- 0.1002, mos = 3.1346 += 0.1800
```
* source: meta-stylespeech; target: stylespeech (ifgsm)
```bash
[ref = 00_gt, test_dir =      01_ori_with_adv] asv = 0.8431 +- 0.0674, mos = 3.5015 += 0.0957
[ref = 00_gt, test_dir =     02_ori_with_base] asv = 0.8511 +- 0.0656, mos = 3.4587 += 0.1007
[ref = 00_gt, test_dir =   03_synthesized_ori] asv = 0.7087 +- 0.0815, mos = 3.3328 += 0.1824
[ref = 00_gt, test_dir =   04_synthesized_adv] asv = 0.5576 +- 0.0996, mos = 3.1476 += 0.2041
[ref = 00_gt, test_dir =  05_synthesized_base] asv = 0.6607 +- 0.1080, mos = 3.2257 += 0.2276
[ref = 00_gt, test_dir = 06_synthesized_black] asv = 0.5610 +- 0.0904, mos = 3.0969 += 0.1984
```
-----------------------------------------------


* White-box attack on meta-stylespeech (ifgsm)
```bash
[ref = 00_ori, test_dir =      01_ori_with_adv] asv = 0.9971 +- 0.0041, mos = 3.4596 += 0.1543
[ref = 00_ori, test_dir =     02_ori_with_base] asv = 0.9945 +- 0.0066, mos = 3.4396 += 0.1458
[ref = 00_ori, test_dir =   03_synthesized_ori] asv = 0.6672 +- 0.0955, mos = 3.1903 += 0.2370
[ref = 00_ori, test_dir =   04_synthesized_adv] asv = 0.5405 +- 0.1092, mos = 3.0785 += 0.2283
[ref = 00_ori, test_dir =  05_synthesized_base] asv = 0.6175 +- 0.1089, mos = 3.1156 += 0.2384
```

* While-box attack on meta-stylespeech (mifgsm)
```bash
[ref = 00_ori, test_dir =      01_ori_with_adv] asv = 0.9809 +- 0.0173, mos = 3.4747 += 0.1409
[ref = 00_ori, test_dir =     02_ori_with_base] asv = 0.9945 +- 0.0066, mos = 3.4416 += 0.1455
[ref = 00_ori, test_dir =   03_synthesized_ori] asv = 0.6672 +- 0.0955, mos = 3.1903 += 0.2370
[ref = 00_ori, test_dir =   04_synthesized_adv] asv = 0.5481 +- 0.0893, mos = 3.1670 += 0.1932
[ref = 00_ori, test_dir =  05_synthesized_base] asv = 0.6143 +- 0.1111, mos = 3.1196 += 0.2320
```

* While-box attack on meta-stylespeech (pgd)
```bash
[ref = 00_ori, test_dir =      01_ori_with_adv] asv = 0.9919 +- 0.0092, mos = 3.5105 += 0.1526
[ref = 00_ori, test_dir =     02_ori_with_base] asv = 0.9945 +- 0.0065, mos = 3.4380 += 0.1404
[ref = 00_ori, test_dir =   03_synthesized_ori] asv = 0.6672 +- 0.0955, mos = 3.1903 += 0.2370
[ref = 00_ori, test_dir =   04_synthesized_adv] asv = 0.5184 +- 0.0970, mos = 3.0843 += 0.1900
[ref = 00_ori, test_dir =  05_synthesized_base] asv = 0.6164 +- 0.1100, mos = 3.1112 += 0.2437
```

### Dongchan Min, Dong Bok Lee, Eunho Yang, and Sung Ju Hwang

This is an official code for our recent [paper](https://arxiv.org/abs/2106.03153).
We propose Meta-StyleSpeech : Multi-Speaker Adaptive Text-to-Speech Generation.
We provide our implementation and pretrained models as open source in this repository.

**Abstract :**
With rapid progress in neural text-to-speech (TTS) models, personalized speech generation is now in high demand for many applications. For practical applicability, a TTS model should generate high-quality speech with only a few audio samples from the given speaker, that are also short in length. However, existing methods either require to fine-tune the model or achieve low adaptation quality without fine-tuning. In this work, we propose StyleSpeech, a new TTS model which not only synthesizes high-quality speech but also effectively adapts to new speakers. Specifically, we propose Style-Adaptive Layer Normalization (SALN) which aligns gain and bias of the text input according to the style extracted from a reference speech audio. With SALN, our model effectively synthesizes speech in the style of the target speaker even from single speech audio. Furthermore, to enhance StyleSpeech's adaptation to speech from new speakers, we extend it to Meta-StyleSpeech by introducing two discriminators trained with style prototypes, and performing episodic training. The experimental results show that our models generate high-quality speech which accurately follows the speaker's voice with single short-duration (1-3 sec) speech audio, significantly outperforming baselines.

Demo audio samples are avaliable [demo page](https://stylespeech.github.io/).

--------
**Recent Updates**
--------
Few modifications on the Variance Adaptor wich were found to improve the quality of the model . 1) We replace the architecture of variance emdedding from one Conv1D layer to two Conv1D layers followed by a linear layer. 2) We add a layernorm and phoneme-wise positional encoding. Please refer to [here](models/VarianceAdaptor.py).


Getting the pretrained models
----------
| Model | Link to the model | 
| :-------------: | :---------------: |
| Meta-StyleSpeech | [Link](https://drive.google.com/file/d/1xGLGt6bK7IapiKNj9YliMBmP5MCBv9OR/view?usp=sharing) |
| StyleSpeech | [Link](https://drive.google.com/file/d/1Q7yLKnFH4UkOjaszikjaovItNAaTyEVN/view?usp=sharing)  |


Prerequisites
-------------
- Clone this repository.
- Install python requirements. Please refer [requirements.txt](requirements.txt)


Inference
-------------
You have to download pretrained models and prepared an audio for reference speech sample.
```bash
python synthesize.py --text <raw text to synthesize> --ref_audio <path to referecne speech audio> --checkpoint_path <path to pretrained model>
```
The generated mel-spectrogram will be saved in `results/` folder.


Preprocessing the dataset
-------------
Our models are trained on [LibriTTS dataset](https://openslr.org/60/). Download, extract and place it in the `dataset/` folder.

To preprocess the dataset : 
First, run 
```bash
python prepare_align.py 
```
to resample audios to 16kHz and for some other preperations.

Second, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
```bash
./montreal-forced-aligner/bin/mfa_align dataset/wav16/ lexicon/librispeech-lexicon.txt  english datset/TextGrid/ -j 10 -v
```

Third, preprocess the dataset to prepare mel-spectrogram, duration, pitch and energy for fast training.
```bash
python preprocess.py
```

Train!
-------------
Train the StyleSpeech from the scratch with
```bash
python train.py 
```

Train the Meta-StyleSpeech from pretrained StyleSpeech with
```bash
python train_meta.py --checkpoint_path <path to pretrained StyleSpeech model>
```


## Acknowledgements
We refered to
* [FastSpeech2](https://arxiv.org/abs/2006.04558)
* [ming024's FastSpeech implementation](https://github.com/ming024/FastSpeech2)
* [Mellotron](https://github.com/NVIDIA/mellotron)
* [Tacotron](https://github.com/keithito/tacotron)
