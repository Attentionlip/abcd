# /bin/python3 /home/gunwoo/gunwoo/Metrics/auto_avsr/preparation/preprocess_lrs2lrs3.py --data-dir "/home/gunwoo/gunwoo/Comparison/LRS3/diffv2s/main"                     --root-dir "/home/gunwoo/gunwoo/Metrics/auto_avsr/preprocessed_diffv2s_lrs3"
# /bin/python3 /home/gunwoo/gunwoo/Metrics/auto_avsr/preparation/preprocess_lrs2lrs3.py --data-dir "/disk2/gunwoo/LipVoicer_attention_vt_3/save_dir/generated_mels/LRS3/w1=1.8_w2=0.5_asr_start=230"   --root-dir "/home/gunwoo/gunwoo/Metrics/auto_avsr/preprocessed_proposed_lrs3_1.8"
# /bin/python3 /home/gunwoo/gunwoo/Metrics/auto_avsr/preparation/preprocess_lrs2lrs3.py --data-dir "/disk2/gunwoo/LipVoicer_attention_vt_3_1.0/save_dir/generated_mels/LRS3/w1=1.0_w2=0.5_asr_start=230"   --root-dir "/home/gunwoo/gunwoo/Metrics/auto_avsr/preprocessed_proposed_lrs3_1.0"
# /bin/python3 /home/gunwoo/gunwoo/Metrics/auto_avsr/preparation/preprocess_lrs2lrs3.py --data-dir "/disk2/gunwoo/LipVoicer_attention_vt_3_1.3/save_dir/generated_mels/LRS3/w1=1.3_w2=0.5_asr_start=230"   --root-dir "/home/gunwoo/gunwoo/Metrics/auto_avsr/preprocessed_proposed_lrs3_1.3"
# /bin/python3 /home/gunwoo/gunwoo/Metrics/auto_avsr/preparation/preprocess_lrs2lrs3.py --data-dir "/disk2/gunwoo/LipVoicer_attention_vt_3_1.3/save_dir/generated_mels/LRS3/w1=5_w2=0.5_asr_start=230"   --root-dir "/home/gunwoo/gunwoo/Metrics/auto_avsr/preprocessed_proposed_lrs3_5.0"
# echo "av_lrs3"
# /bin/python3 /home/gunwoo/gunwoo/Metrics/auto_avsr/eval_1.py data.dataset.root_dir="/home/gunwoo/gunwoo/Metrics/auto_avsr/preprocessed_proposed_lrs3_5.0"

# /bin/python3 /home/gunwoo/gunwoo/Metrics/auto_avsr/preparation/preprocess_lrs2lrs3.py --data-dir "/home/gunwoo/gunwoo/LipVoicer_attention_vt_1/save_dir/generated_mels/LRS2/w1=1.8_w2=0.5_asr_start=230"   --root-dir "/home/gunwoo/gunwoo/Metrics/auto_avsr/preprocessed_proposed_lrs2_vt1"
# echo "proposed_10"
# /bin/python3 /home/gunwoo/gunwoo/Metrics/auto_avsr/eval_1.py data.dataset.root_dir="/home/gunwoo/gunwoo/Metrics/auto_avsr/preprocessed_proposed_lrs2_vt1"




# # /bin/python3 /home/gunwoo/gunwoo/Metrics/auto_avsr/preparation/preprocess_lrs2lrs3.py --data-dir "/home/gunwoo/gunwoo/LipVoicer_attention_vt_2/save_dir/generated_mels/LRS2/w1=1.8_w2=0.5_asr_start=230"   --root-dir "/home/gunwoo/gunwoo/Metrics/auto_avsr/preprocessed_proposed_lrs2_vt2"
# echo "proposed_4"
# /bin/python3 /home/gunwoo/gunwoo/Metrics/auto_avsr/eval_1.py data.dataset.root_dir="/home/gunwoo/gunwoo/Metrics/auto_avsr/preprocessed_proposed_lrs2_vt2"





/bin/python3 /home/gunwoo/gunwoo/LipVoicer_attention_LoRA/inference_full_test_split_1.py generate.w_video=-1
/bin/python3 /home/gunwoo/gunwoo/LipVoicer_attention_LoRA/inference_full_test_split_1.py generate.w_video=0.5
# /bin/python3 /home/gunwoo/gunwoo/LipVoicer_attention_LoRA/inference_full_test_split_1.py generate.w_video=1.0
# /bin/python3 /home/gunwoo/gunwoo/LipVoicer_attention_LoRA/inference_full_test_split_1.py generate.w_video=1.5
# /bin/python3 /home/gunwoo/gunwoo/LipVoicer_attention_LoRA/inference_full_test_split_1.py generate.w_video=2.0
# /bin/python3 /home/gunwoo/gunwoo/LipVoicer_attention_LoRA/inference_full_test_split_1.py generate.w_video=3.0
# /bin/python3 /home/gunwoo/gunwoo/LipVoicer_attention_LoRA/inference_full_test_split_1.py generate.w_video=4.0


