import os
import json

with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)

class LIDConfig(object):

    dir = config['dataDir']['dir']

    train_path = config['dataDir']['train_path'].replace('$dir', dir)

    test_path = config['dataDir']['test_path'].replace('$dir', dir)

    val_path = config['dataDir']['val_path'].replace('$dir', dir)

    batch_size = int(config['model_parameters']['batch_size'])
    epochs = int(config['model_parameters']['epochs'])

    model_type = 'UpstreamTransformer'
    
    # upstream model to be loaded from s3prl. Some of the upstream models are: wav2vec2, hubert, TERA, mockingjay etc.
    #See the available models here: https://github.com/s3prl/s3prl/blob/master/s3prl/upstream/README.md
    upstream_model = config['model_parameters']['upstream_model']

    # number of layers in encoder (transformers)
    num_layers = 6

    # feature dimension of upstream model. For example, 
    # For wav2vec2, feature_dim = 768
    # For npc, feature_dim = 512
    # For tera, feature_dim = 768
    feature_dim = 768

    # No of GPUs for training and no of workers for datalaoders
    gpu = int(config['gpu'])
    n_workers = int(config['n_workers'])

    # model checkpoint to continue from
    model_checkpoint = None
    
    # LR of optimizer
    lr = float(config['model_parameters']['lr'])

    run_name = model_type
