# Text-classification model based on LSTMs
This repository contains an example of Japanese text classification based on LSTMs.

## Requirements
- python3
- gensim 3.8.3 or later
- mecab-python3 0.996.1
- numpy 1.16.2 or later
- pytorch 1.5.0 or later
- scikit-learn 0.22.1 or later

## Usage
### Preprocessing
```
sh ./download_data.sh
python main.py preprocess -d text --embedding-path jawiki.all_vectors.300d.txt
```

## Training
```
python main.py train --embedding-path embedding.bin --embedding-size 50 --hidden-size 400 --max-epochs 3 --weight-dropout 0.2 --num-layers 1
```

## Evaluation
```
python main.py evaluate
```

## Author
Takuya Makino takuyamakino15@gmail.com
