# Meta-StyleSpeech : Multi-Speaker Adaptive Text-to-Speech Generation

### Eval results (eval on VCTK)

* White-box attack on meta-stylespeech (ifgsm)
```bash
[ref = 00_gt, test_dir =      01_ori_with_adv] asv = 0.8544 +- 0.0620, mos = 3.4652 += 0.1338                                                                                                               
[ref = 00_gt, test_dir =     02_ori_with_base] asv = 0.8511 +- 0.0658, mos = 3.4470 += 0.1270                                                                                                               
[ref = 00_gt, test_dir =   03_synthesized_ori] asv = 0.7087 +- 0.0815, mos = 3.2460 += 0.2281                                                                                                               
[ref = 00_gt, test_dir =   04_synthesized_adv] asv = 0.5709 +- 0.0979, mos = 3.0750 += 0.2342                                                                                                               
[ref = 00_gt, test_dir =  05_synthesized_base] asv = 0.6633 +- 0.1076, mos = 3.1587 += 0.2395                                                                                                               
```

* While-box attack on meta-stylespeech (mifgsm)
```bash
[ref = 00_gt, test_dir =      01_ori_with_adv] asv = 0.8419 +- 0.0682, mos = 3.4853 += 0.1234
[ref = 00_gt, test_dir =     02_ori_with_base] asv = 0.8511 +- 0.0656, mos = 3.4485 += 0.1279
[ref = 00_gt, test_dir =   03_synthesized_ori] asv = 0.7087 +- 0.0815, mos = 3.2460 += 0.2281
[ref = 00_gt, test_dir =   04_synthesized_adv] asv = 0.5800 +- 0.0929, mos = 3.1694 += 0.1922
[ref = 00_gt, test_dir =  05_synthesized_base] asv = 0.6632 +- 0.1067, mos = 3.1625 += 0.2361
```

* While-box attack on meta-stylespeech (pgd)
```bash
[ref = 00_gt, test_dir =      01_ori_with_adv] asv = 0.8497 +- 0.0647, mos = 3.5156 += 0.1316
[ref = 00_gt, test_dir =     02_ori_with_base] asv = 0.8511 +- 0.0655, mos = 3.4460 += 0.1237
[ref = 00_gt, test_dir =   03_synthesized_ori] asv = 0.7087 +- 0.0815, mos = 3.2460 += 0.2281
[ref = 00_gt, test_dir =   04_synthesized_adv] asv = 0.5527 +- 0.0996, mos = 3.0957 += 0.1932
[ref = 00_gt, test_dir =  05_synthesized_base] asv = 0.6647 +- 0.1070, mos = 3.1602 += 0.2412
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
