# Flow Graphs - Linguistic representations for fewer-shot relation extraction across domains in procedural text

## Overview

This repository contains the code for investigating whether linguistic representations, in the form of dependency and [AMR](amr.isi.edu) parses improve the few-shot performance of relation extraction models on procedural text. 

### Conventions

This repo relies on a few libraries that define its style. Knowing which ahead of time is helpful to understand how the code is written. The major libraries are:

- [**Fire**](https://google.github.io/python-fire/): This library allows the creation of command-line applications directly from python objects, in our case largely functions. Modules in this repo usually contain a call to `fire.Fire(<function>)`, which converts the function passed as the argument into a command line program. Arguments to the command line program are automatically created based on the argument names and types of the function; especially for small programs, this provides a much neater interface than `argparse`. 

- [**DVC**](https://dvc.org): We manage our data with DVC. Metadata to data and model checkpoints are stored in this repository, and the corresponding data can be pulled, versioned, and inspected using the DVC commnand line tool. For more information, see the DVC section of this document.

## Data Preprocessing

All data preprocessing code is located in the `preprocess` package. The code to preprocess data in this package generally takes files in a data directory, `/data/<dataset_name>`, and creates a new file or files in that directory. Data preprocessing for the experiments in this repository is performed in the following steps:

**Data Standardization**: In this stage, we take datasets in their original formats, and standardize their relation extraction content into a standard json format. Each relation, which is defined as a triple of `(entity1, relation, entity2)` will be given the following format:

```json
{
    "_id": "The index or UID of the relation",
    "arg1_start": "The string index of the first entity's first character",
    "arg1_end": "The string index of the first entity's last character",
    "arg1_word": "The string corresponding to the first entity",
    "arg1_label": "The entity type for the first entity",
    "arg2_start": "The string index of the second entity's first character",
    "arg2_end": "The string index of the second entity's last character",
    "arg2_word": "The string corresponding to the second entity",
    "arg2_label": "The entity type for the second entity",
    "arg_label": "The label for the relation",
}
```
The code to accomplish this standardization is defined in `preprocess/standardize_dataset_format.py`, and can be run as a script with a single argument, the name of the dataset to standardize, e.g. 

```
python -m preprocess.standardize_dataset_format --dataset risec
```

**AMR Generation**: In this stage, we annotate the corpora with AMR parses. This step is done separately from the next step, in which we align and add dependency parses, because AMR parses are comparitively expensive to generate. From this, we generate files in the data folder corresponding to each split: `amr_<split>.pkl`. This step is currently defined in the `amr/annotate_datasets.py` script, but will be moved into the `preprocess` package. 

**Aligning and adding parses**: In this stage, we align the original annotations to BERT token representation for the baseline model; we also perform dependency parses for each sentence, and align the original annotations with nodes in both the dependency and AMR graph representations for the interventions. We then format the tokenized representations as well as both graph representations into a dataset, which is saved as `data_amr.dill` in the corresponding data folder. This is defined in the `preprocess/align_and_add_parses.py` script.

### Fire Command Group

Both of the preprocess scripts currently in the `preprocess` module can also be called from the package level, e.g.:
```python -m preprocess align_and_add_parses --dataset risec```

### DVC

The data for this project is managed and versioned by [DVC](dvc.org), and it is stored in [this Google Drive folder](https://drive.google.com/drive/u/1/folders/1Ql9-M5k4fjjR5VQPxrVaQ8UJ2gS1HLf3).

You can find instructions for installing DVC [here](https://dvc.org/doc/install). Once you have DVC installed, run `dvc pull` from the root of the repo. This will pull down all the files that have been checked into DVC thus far. 

DVC works in a similar fashion to [git-lfs](https://git-lfs.github.com/): it stores pointers and metadata for your data in the git repository, while the files live elsewhere (in this case, on Google Drive). As you work with data, such as in [the DVC tutorial](https://dvc.org/doc/start/data-and-model-versioning), DVC will automatically add the files you have tracked with it to the `.gitignore` file, and add new `.dvc` files that track the metadata associated with those files. 

#### Sample Workflow

- **Pull data down**: run `dvc pull` to pull down the data file into the repository folder
- **Modify your data**: as you would without DVC, use, modify, and work with your data.
- **Add new/modified data to DVC**: using `dvc add ...` in a similar fashion to a `git add`, add your new or modified data files to DVC
- **Add the corresponding metadata to git**: Once the data file has been added to DVC, a corresponding `.dvc` file will have been created. Add or update this into git, then push. 
- **Sync the locally updated DVC data with the remote**: finally, push the data itself up to Google Drive with the `dvc push` command. 

tl;dr:
- dvc pull
- dvc add <data_file>
- git add/commit <data_file.dvc>
- git push
- dvc push

## Training Models

There are two paradigms for training models in this repository: in-domain and transfer. In the in-domain case, which is specified in `train_single_domain_model.py`, we train and evaluate a model on a single dataset; in the transfer case, we take a model which has been trained in-domain on a dataset, and transfer it in the few-shot setting to a different dataset. The transfer case is defined in `train_transfer_model.py`. Both of these scripts take only parameters necessary to specify which model to train, and device-specific details like the GPU index. All other parameters and hyperparameters, such as the base model, whether to use either graph representation, and how to log details to Weights and Biases, are specified in the `experiment_configs.py` file. Valid experiment configs are the keys in the `model_configurations` dictionary.

In the indomain case, the training script only requires three parameters: the dataset on which to train, the seed to use, and an experiment configuration. The transfer case takes the source and target domains, the number or fraction of examples to use for fewshot training, a seed, and an experiment configurations. We save the best model after early stopping with a 5-epoch patience, and those models are saved in the `checkpoints/` directory, with names specified based on the parameters used. Code that generates the names of models can be found in `modeling/metadata_utils.py`

## Evaluating Models

To evaluate already-trained models, use the `eval_transfer_model.py` and `eval_indomain_model.py` scripts. Because of idiosyncracies in pytorch model loading, you will need to specify the model configuration that a model was trained with.

## Scripts, singularity, etc. 

TK.
