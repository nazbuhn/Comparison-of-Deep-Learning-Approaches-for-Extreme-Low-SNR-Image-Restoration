### Self-Inspired Noise2Noise

This folder contains a copy of the authors' implementation of SN2N at https://github.com/SR-Wiki/SN2N .

The script `split_and_normalize.py` will split a dataset into train/test splits and normalize the images using percentage normalization.

    python split_and_normalize.py \
        --path <path to dataset root>
        --dataset 01
        --outpath dataset

Then run SN2N as follows:

    python -m scripts.Script_SN2N_datagen_2D \
          --img_path dataset/01 \
          --P2Pmode 0 --P2Pup 0 --BAmode 2 --SWsize 64

    python -m scripts.Script_SN2N_trainer_2D  \
          --img_path dataset/01/datasets \
          --sn2n_loss 1 --bs 32 --lr 2e-4 --epochs 100

    python -m scripts.Script_SN2N_inference_2D \
          --img_path dataset/01/test \
          --model_path dataset/01/models \
          --infer_mode 1

To use one of the pre-trained models from GigaDB, place the cropped images in the `test/` directory and checkpoint file the `models/` directory.
