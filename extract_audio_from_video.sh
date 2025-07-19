# 1. Extract the audio files from the videos (audio files will be saved in a WAV format)

# python3 dataloaders/extract_audio_from_video.py --ds_dir /mnt/data/datasets/lrs2/mvlrs_v1 \
#                                                --split main  \
#                                                --out_dir data


# python3 dataloaders/extract_audio_from_video.py --ds_dir /mnt/data/datasets/lrs2/mvlrs_v1 \
#                                                --split pretrain  \
#                                                --out_dir data


# 2. Compute the log mel-spectrograms and save them
# Before excute the wav2mel.py, please modify the configs.config.ymal file.

# python3 dataloaders/wav2mel.py


# 3. 