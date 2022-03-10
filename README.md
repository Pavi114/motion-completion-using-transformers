# Transformer based Motion In-betweening

This repository contains the code accompaniying the thesis project "Transformer based Motion In-betweening".

## TODO

- [x] Data Preprocessing 
    - [x] `util/convert.py`
    - [x] `util/load_data.py`
    - [x] `util/linear_interpolation.py`
- [x] Transformer Models 
    - [x] `model/transformer.py`
    - [x] `model/encoding/positional_encoding.py`
- [ ] Loss and Metrics
    - [x] L1 Loss
    - [ ] L2 Losses
    - [ ] NPSS
- [x] Training Script
- [ ] Evaluation Script

## Downloading Data

### LAFAN1 Dataset

- Download the dataset from [Ubisoft's Github Repository](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) and extract it to `/data/lafan1/`

## Installation

1. Install Pre-Requisites 

    - Python 3.9
    - PyTorch 1.10

2. Clone the repository
    ```git clone https://github.com/Pavi114/motion-completion-using-transformers```

3. Copy config/default.yml to config/`model_name`.yml and edit as needed.

4. Install Python Dependencies

    - Create a virtualenv: `python3 -m virtualenv -p python3.9 venv`

    - Install Dependencies: `pip install -r requirements.txt`

## Execution

First activate the venv: `source venv/bin/activate`

### Training

```
train.py [-h] [--model_name MODEL_NAME] [--save_weights | --no-save_weights] [--load_weights | --no-load_weights]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Name of the model. Used for loading and saving weights.
  --save_weights, --no-save_weights
                        Save model weights. (default: False)
  --load_weights, --no-load_weights
                        Load model weights. (default: False)
```

### Visualization

0. Navigate to `./viz` directory

    ```
    cd ./viz
    ```

1. Install NPM Modules

    ```
    npm install
    ```

2. Build visualizer

    ```
    npm run build
    ```

3. Copy output file to `./dist`

    ```
    cp output/[MODEL_NAME] viz/dist/static/animations/[MODEL_NAME]
    ```

4. Run viz

    ```
    npm start
    ```

### Evaluation

```insert evaluation command here```





