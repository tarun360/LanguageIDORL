import os

class LIDConfig(object):
    # path to the unzuipped TIMIT data folder
    train_path = '/notebooks/LID/LanguageIDORL/data/train/wav.scp'

    test_path = '/notebooks/LID/LanguageIDORL/data/dev_all/wav.scp'

    val_path = '/notebooks/LID/LanguageIDORL/data/dev_all/wav.scp'

    batch_size = 4
    epochs = 200

    # model type
    ## AHG 
    # wav2vecTransformer
    
    ## H
    # wav2vecTransformer
    model_type = 'UpstreamTransformer'

    # RMSE, UncertaintyLoss
    loss = "UncertaintyLoss"
    
    ## H
    # wav2vecTransformer
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
    gpu = '-1'
    n_workers = 0

    # model checkpoint to continue from
    model_checkpoint = None
    
    # LR of optimizer
    lr = 1e-4

    run_name = model_type
