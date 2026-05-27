### Content-Aware Image Restoration (CARE)

This folder contains scripts to train and test a CARE model using the authors' implementation of CARE at https://github.com/csbdeep/csbdeep .

To train the model:

    python train_care_generator.py \
          --path <path to crops directory> \
          --dataset <name of dataset e.g. 01>

To run inference with a trained model or a pre-trained checkpoint from GigaDB:

    python test_care.py \
          --path <path to crops directory> \
          --dataset <name of dataset e.g. 01> \
          --checkpoint <path to checkpoint folder>

