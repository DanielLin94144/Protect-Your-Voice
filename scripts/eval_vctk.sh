#!/bin/bash

DATA_ROOT="/hdd0/CORPORA/VCTK-Corpus"

EXP_NAME="vctk_white_box_ifgsm"
echo "EXP_NAME = ${EXP_NAME}"
python3 eval_attack.py --data_root $DATA_ROOT --checkpoint_path pretrained_model/meta_stylespeech.pth.tar --save_dir results/${EXP_NAME}
python3 eval_agent.py results/${EXP_NAME}

EXP_NAME="vctk_white_box_mifgsm"
echo "EXP_NAME = ${EXP_NAME}"
python3 eval_attack.py --data_root $DATA_ROOT --checkpoint_path pretrained_model/meta_stylespeech.pth.tar --save_dir results/${EXP_NAME} --momentum 1
python3 eval_agent.py results/${EXP_NAME}

EXP_NAME="vctk_white_box_pgd"
echo "EXP_NAME = ${EXP_NAME}"
python3 eval_attack.py --data_root $DATA_ROOT --checkpoint_path pretrained_model/meta_stylespeech.pth.tar --save_dir results/${EXP_NAME} --random_start
python3 eval_agent.py results/${EXP_NAME}

