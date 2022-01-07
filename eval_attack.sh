

python eval_attack.py --checkpoint_path pretrained_model/meta_stylespeech.pth.tar \
                    --data_root /home/daniel094144/Daniel/data/VCTK-Corpus \
                    --save_dir results_cat_no300_both \
                    --random_start \
                    --learning_rate 0.001 \
                    --blackbox_target_path pretrained_model/stylespeech.pth.tar \

