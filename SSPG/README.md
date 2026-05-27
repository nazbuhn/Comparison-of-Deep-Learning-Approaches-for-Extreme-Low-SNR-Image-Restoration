### Self-Supervised Poisson Gaussian

This folder contains a copy of the [authors' implementation of SSPG at https://github.com/jonathanventura/self-supervised-poisson-gaussian](https://github.com/jonathanventura/self-supervised-poisson-gaussian).

To train the model:

    python train_sspg.py \
          --path <path to crops directory> \
          --dataset <name of dataset e.g. 01>

To run inference with a trained model or a pre-trained checkpoint from GigaDB:

    python test_sspg.py \
          --path <path to crops directory> \
          --dataset <name of dataset e.g. 01> \
          --checkpoint <path to checkpoint file>

