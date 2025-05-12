<!-- >📋  A template README.md for code accompanying a Machine Learning paper -->

# SF$^2$Bench: Evaluating Data-Driven Models for Compound Flood Forecasting in South Florida

This repository is the official implementation of [SF$^2$Bench](https://arxiv.org/abs/Placeholder). A benchmark paper of compound flood in the South Florida area. In this paper, we consider seveal key factors for compound flood forecasting, including sea level, rainfall, groundwater and human control/management activities.

<!-- >📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

## Dataset
Please download the dataset from [Placeholder]() and Unzip to dataset folder. The detail information of each file is provided in [Dataset.md](./dataset/Dataset.md)


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 





# South Florida Flood dataset & Forcasting Benchmark

This dataset is a collection of South Florida water stage dataset.

--------------------------------------------
## dataset download

- Onedrive
- trustai4s 

--------------------------------------------
## development

- enviroment based on the [NeuralForecast](https://github.com/Nixtla/neuralforecast/tree/main?tab=readme-ov-file)
- follow the src/demo pipeline
- complete the model file and put in folder in src/models
- create your own test file and test

> Please make sure every model have the same apis for train and evaluation. We will need one file to run all the models.

--------------------------------------------
## run & evaluate 

- set SEED = 2025 , to make sure reproducibility 
- use adamW as default optimizer
- use the default hyperparmeters according to the paper or their code 
- record the running command and put it into run.sh 

> Please make sure the commands are recorded. We will need them for reproduction.


--------------------------------------------
## Git

- (init code)git clone / git pull 
- (new local branch) git branch [branch-name]
- (switch to new branch) git checkout [branch-name]
- (work, add your code)  git add 
- (commit) git commit 
- (switch to main branch) git checkout [main]
- (update main branch) git pull
- (merge local branch to main) git merge [branch-name]
- (push to server)git push

> If you already have a local branch, first update main, then merge main to local branch. 


