import numpy as np
import argparse
import toml

from .train import main

if __name__ == '__main__':
    np.random.seed(0)

    parser = argparse.ArgumentParser()

    # Training parameters.
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training.')
    parser.add_argument('--sample_size', type=int, default=2, help='Number of positive pair to sample.')
    parser.add_argument('--epochs', type=int, default=20, help='Max number of epochs to train.')
    parser.add_argument('--num_neg_sample', type=int, default=10, help='Number of negative example to sample.')
    parser.add_argument('--margin', type=float, default=0.05, help='Margin.')
    parser.add_argument('--im_loss_factor', type=float, default=1.5,
                        help='Factor multiplied with image loss. Set to 0 for single direction.')
    parser.add_argument('--sent_only_loss_factor', type=float, default=0.05,
                        help='Factor multiplied with sent only loss. Set to 0 for no neighbor constraint.')
    parser.add_argument("--image_feat_path", type=str, help="Path to mat file of img feats")
    parser.add_argument("--sent_feat_path", type=str, help="Path to mat file of sent fets")
    # Optional config.toml file
    parser.add_argument('--config',type=str,default=None,help='Optional config file to set parameters')

    args = parser.parse_args()
    config = vars(args)

    if args.config:
        with open(args.config) as f:
            config.update(toml.load(f))


    main(config)