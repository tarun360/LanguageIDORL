# Language ID

This Repository contains the code for language identification from speech utterance. This repository uses [s3prl](https://github.com/s3prl/s3prl) library to load various upstream models like wav2vec2, CPC, TERA etc. This repository uses TIMIT dataset. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages for preparing the dataset, training and testing the model.

```bash
pip install -r requirements.txt
```

## Usage


### Update Config and Logger
Update the config.py file to update the upstream model, batch_size, gpus, lr, etc and change the preferred logger in train_.py files

### Training
```bash
./run.sh
```

### Testing
```bash
./run_test.sh
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Reference
- [1] S3prl: The self-supervised speech pre-training and representation learning toolkit. AT Liu, Y Shu-wen

