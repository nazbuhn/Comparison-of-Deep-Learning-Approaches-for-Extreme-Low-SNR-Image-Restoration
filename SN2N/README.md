### Self-Inspired Noise2Noise

We used the [authors' implementation of SN2N at https://github.com/SR-Wiki/SN2N](https://github.com/SR-Wiki/SN2N).

The script `split_and_normalize.py` will split a dataset into train/test splits and normalize the images using percentage normalization.

    python split_and_normalize.py \
        --path <path to dataset root>
        --dataset <dataset name e.g. 01>
        --outpath <output path to preprocessed dataset>

We then ran SN2N as follows:

    python -m scripts.Script_SN2N_datagen_2D \
          --img_path <path to preprocessed dataset>/train \
          --P2Pmode 0 --P2Pup 0 --BAmode 2 --SWsize 64

    python -m scripts.Script_SN2N_trainer_2D  \
          --img_path <path to preprocessed dataset>/train \
          --sn2n_loss 1 --bs 32 --lr 2e-4 --epochs 100

    python -m scripts.Script_SN2N_inference_2D \
          --img_path <path to preprocessed dataset>/test \
          --model_path <path to preprocessed dataset>/models \
          --infer_mode 1

