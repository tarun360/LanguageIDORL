import os
import json

with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)

class LIDConfig(object):
    train_path = config['dir']['train_path']

    test_path = config['dir']['test_path']

    val_path = config['dir']['val_path']

    batch_size = int(config['parameters']['batch_size'])
    epochs = int(config['parameters']['epochs'])

    model_type = 'UpstreamTransformer'
    
    # upstream model to be loaded from s3prl. Some of the upstream models are: wav2vec2, TERA, mockingjay etc.
    #See the available models here: https://github.com/s3prl/s3prl/blob/master/s3prl/upstream/README.md
    upstream_model = 'wav2vec2'

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
    lr = float(config['parameters']['lr'])

    run_name = model_type
