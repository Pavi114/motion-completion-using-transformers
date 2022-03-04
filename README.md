# Transformer based Motion In-betweening

This repository contains the code accompaniying the thesis project "Transformer based Motion In-betweening".

## TODO

- [ ] Data Preprocessing 
    - [ ] `util/convert.py`
    - [x] `util/load_data.py`
    - [ ] `util/linear_interpolation.py`
- [x] Transformer Models 
    - [x] `model/transformer.py`
    - [x] `model/encoding/positional_encoding.py`
- [ ] Loss and Metrics (IK)
    - [ ] L1 Loss
    - [ ] L2 Losses
    - [ ] NPSS
- [ ] Training Script
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

3. Copy hyperparameters.example.py to hyperparameters.py
    ```cp hyperparameters.example.py hyperparameters.py```

4. Install Python Dependencies

    ```pip install -r requirements.txt```

## Execution

### Training

```insert training command here```

### Evaluation

```insert evaluation command here```





