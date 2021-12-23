#!/bin/bash

EXP_NAME="vctk_white_box"

echo "EXP_NAME = ${EXP_NAME}"
python3 eval_attack.py --checkpoint_path pretrained_model/meta_stylespeech.pth.tar --save_dir results/${EXP_NAME}
python3 eval_agent.py results/${EXP_NAME}

