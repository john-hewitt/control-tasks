<p align="center">
  <img src="doc-assets/confounder2.png" width="700" title="hover text" alt="The probe confounder problem: what causes high probe accuracy?">
</p>


# control-tasks
Repository describing example random control tasks for designing and interpreting neural probes.
Control tasks complement linguistic tasks when probing neural models by helping researchers come to an understanding of the ability of a probe to memorize, for each unique word, a randomly chosen output -- and apply this output regardless of context.

Based on the paper [Designing and Interpreting Probes with Control Tasks](#).

See the blog post on control tasks for a brief introduction.

This repo is a fork of the [structural probes](https://github.com/john-hewitt/structural-probes) codebase. Added are:

 - part-of-speech tagging task and families of probes (linear, MLP).
 - Random control task for part-of-speech tagging.
 - dependency edge prediction task and families of probes (bilinear, MLP)
 - Random control task for dependency edge prediction.
 - Options to use dropout, weight decay, matrix rank constraints, hidden state size constraints, and limited numbers of gradient steps for regularizing probes.

## Installing & Getting Started

1. Clone the repository.

        git clone https://github.com/john-hewitt/control-tasks/
        cd structural-probes
        
1. [Optional] Construct a virtual environment for this project. Only `python3` is supported.
 
        virtualenv sp-env
        source activate sp-env

1. Install the required packages. This mainly means `pytorch`, `scipy`, `numpy`, `seaborn`, etc.

        pip install -r requirements.txt
        
1. Download some pre-packaged data from the [English Universal Dependencies (EWT) dataset](https://universaldependencies.org/treebanks/en_ewt/index.html) to get your feet wet.

        bash ./download_example.sh
  
    This will make the directory `example/data`, and in it will be 9 files, 3 for each of train,dev,test.
     - `en_ewt-ud-{train,dev,test}.conllu`: the parsed language data
     - `en_ewt-ud-{train,dev,test}.txt`: whitespace-tokenized, sentence-per-line language data.
     - `en_ewt-ud-{train,dev,test}.elmo-layers.hdf5`: the ELMo hidden states for each sentence of the language data, constructed by running elmo on the `.txt` files.

1. Run an experiment using an example experiment configuration for a linguistic task, as well as a control task, and take a look at the resultant reporting!

        python control-tasks/run_experiment.py example/config/en_ewt-pos-corrupted0-rank1000-0hid-ELMo1.yaml
        python control-tasks/run_experiment.py example/config/en_ewt-pos-corrupted1-rank1000-0hid-ELMo1.yaml
        
   The path to a new directory containing the results of the experiment will be in the first few lines of the logging output of the script. Once you're there, you'll see `dev.label_acc`, reporting the labeling accuracy for the experiment.
   We'll go over this later, but `corrupted0` or `c0` means that the task is a linguistic task, and `corrupted1` or `c1` means the task is a control task. 

## The experiment config file
Experiments run with this repository are specified via `yaml` files that completely describe the experiment (except the random seed.)
In this section, we go over each top-level key of the experiment config.

### Dataset:
 - `observation_fieldnames`: the fields (columns) of the conll-formatted corpus files to be used.
   Must be in the same order as the columns of the corpus.
   Each field will be accessable as an attribute of each `Observation` class (e.g., `observation.sentence`
   contains the sequence of tokens comprising the sentence.)
 - `corpus`: The location of the train, dev, and test conll-formatted corpora files. Each of `train_path`,
   `dev_path`, `test_path` will be taken as relative to the `root` field.
 - `embeddings`: The location of the train, dev, and test pre-computed embedding files (ignored if not applicable.
 Each of `train_path`, `dev_path`, `test_path` will be taken as relative to the `root` field.
        - `type` is ignored.
 - `batch_size`: The number of observations to put into each batch for training the probe. 20 or so should be great.
 - `dataset_size`: The number of observations to cap the _training_ data to when training this probe. 
```
dataset:
  observation_fieldnames:
     - index
     - sentence
     - lemma_sentence
     - upos_sentence
     - xpos_sentence
     - morph
     - head_indices
     - governance_relations
     - secondary_relations
     - extra_info
     - embeddings
  corpus:
    root: example/data/en_ewt-ud-sample/
    train_path: en_ewt-ud-train.conllu
    dev_path: en_ewt-ud-dev.conllu
    test_path: en_ewt-ud-test.conllu
  embeddings:
    type: token #{token,subword}
    root: example/data/en_ewt-ud-sample/ 
    train_path: en_ewt-ud-train.elmo-layers.hdf5
    dev_path: en_ewt-ud-dev.elmo-layers.hdf5
    test_path: en_ewt-ud-test.elmo-layers.hdf5
  batch_size: 40
  dataset_size: 40000
```
### Model
 - `hidden_dim`: The dimensionality of the representations to be probed.
    The probe parameters constructed will be of shape (hidden_dim, maximum_rank)
 - `embedding_dim`: ignored
 - `model_type`: One of `ELMo-disk`, `BERT-disk`, `ELMo-decay`, `ELMo-random-projection` as of now. 
   Used to help determine which `Dataset` class should be constructed, as well as which model will construct the representations for the probe.
   The `Decay0` and `Proj0` baselines in the paper are from `ELMo-decay` and `ELMo-random-projection`, respectively.
   In the future, will be used to specify other PyTorch models.
 - `use_disk`: Set to `True` to assume that pre-computed embeddings should be stored with each `Observation`; Set to `False` to use the words in some downstream model (this is not supported yet...)
 - `model_layer`: The index of the hidden layer to be used by the probe. For example, `ELMo` models can use layers `0,1,2`; BERT-base models have layers `0` through `11`; BERT-large `0` through `23`.
```
model:
  hidden_dim: 1024 # ELMo hidden dim
  model_type: ELMo-disk # BERT-disk, ELMo-disk,
  use_disk: True
  model_layer: 1 # BERT-base: {1,...,12}; ELMo: {1,2,3}
```

### Probe, probe-training
 - `task_signature`: Specifies the function signature of the task.
 Currently, can be either `word_label`, for part-of-speech tagging tasks; or `word_pair_label` for dependency edge prediction tasks.
 - `task_name`: A unique name for each task supported by the repository. Right now, this includes `corrupted-part-of-speech` (for part-of-speech tagging and its control task) and `corrupted-edge-labels` for dependency edge prediction and its control task.
 - `maximum_rank`: Specifies the dimensionality of the space to be projected into, if `psd_parameters=True`.
   The projection matrix is of shape (hidden_dim, maximum_rank).
   The rank of the subspace is upper-bounded by this value.
   If `psd_parameters=False`, then this is ignored.
 - `diagonal`: Ignored.
 - `hidden_layers`: Number of hidden layers in the probe network, for part-of-speech tagging and its control task. Marking `0` means a linear model; `1` means an MLP with one hidden layer, `2` an MLP with 2 hidden layers.
 - `dropout`: Dropout percent to be applied at the input embeddings and at any hidden layer during training.
 - `probe_spec`: Specification of probe parameters for the dependency edge prediction task and its control task. `MLP` for the probe type gives a multi-layer perceptron, in which case the number of hidden layers (`1` or `2`) is specified by `probe_hidden_layers`. If `probe_type` is set to `bilinear`, a bilinear probe is used.
 - `corrupted_token_percent`: The percent of tokens' outputs in the data to replace with control task outputs. Should be set to `0` for a linguistic task, or `1` for a control task. Was previously used, given values in `(0,1)`, to make mixture tasks, which you can try out if you'd like!
 - `params_path`: The path, relative to `args['reporting']['root']`, to which to save the probe parameters.
 - `epochs`: The maximum number of epochs to which to train the probe. 
   (Regardless, early stopping is performed on the development loss.)
 - `loss`: A string to specify the loss class. Right now, `cross-entropy` is available for labeling tasks.
    The class within `loss.py` will be specified by a combination of this and the task name.
 - `weight_decay`: Weight decay (L2 regularization) to be applied during training.
```
probe:
  task_signature: word_label # word, word_pair
  task_name: corrupted-part-of-speech
  maximum_rank: 1000
  psd_parameters: True
  diagonal: False
  hidden_layers: 0
  dropout: 0
  params_path: predictor.params
  misc:
    corrupted_token_percent: 0.0
  probe_spec:
    probe_type: MLP
    probe_hidden_layers: 1
probe_training:
  epochs: 40
  loss: cross-entropy
  weight_decay: 0.0
```

### Reporting
 - `root`: The path to the directory in which a new subdirectory should be constructed for the results of this experiment.
 - `observation_paths`: The paths, relative to `root`, to which to write the observations formatted for quick reporting later on.
 - `prediction_paths`: The paths, relative to `root`, to which to write the predictions of the model.
 - `reporting_methods`: A list of strings specifying the methods to use to report and visualize results from the experiment.
    Dependency edge prediction and its control task use `uuas` to report accuracy. When reporting `uuas`, some `tikz-dependency` examples are written to disk as well.
    Part-of-speech tagging and its control task use `label_acc` to report accuracy.
    
```
reporting:
  root: example/results
  observation_paths:
    train_path: train.observations
    dev_path: dev.observations
    test_path: test.observations
  prediction_paths:
    train_path: train.predictions
    dev_path: dev.predictions
    test_path: test.predictions
  reporting_methods:
    - label_acc 
    - uuas
```


## Experiments on new datasets or models
Right now, the official way to run experiments on new datasets and representation learners is:

1. Have a `conllx` file for the train, dev, and test splits of your dataset.
1. Write contextual word representations to disk for each of the train, dev, and test split in `hdf5` format, where the index of the sentence in the `conllx` file is the key to the `hdf5` dataset object. That is, your dataset file should look a bit like `{'0': <np.ndarray(size=(1,SEQLEN1,FEATURE_COUNT))>, '1':<np.ndarray(size=(1,SEQLEN1,FEATURE_COUNT))>...}`, etc. Note here that `SEQLEN` for each sentence must be the number of tokens in the sentence as specified by the `conllx` file.
1. Edit a `config` file from `example/config` to match the paths to your data, as well as the hidden dimension and labels for the columns in the `conllx` file. Look at the experiment config section of this README for more information therein. One potential gotcha is that you _must_ have an `xpos_sentence` field in your conllx (as labeled by your yaml config) since this will be used at evaluation time. 

## Citation

If you use this repository, please cite:

      @InProceedings{hewitt2019designing,
        author =      "Hewitt, John and Liang, Percy",
        title =       "Designing and Interpreting Probes with Control Tasks",
        booktitle =   "Conference on Empirical Methods in Natural Language Processing",
        year =        "2019",
        publisher =   "Association for Computational Linguistics",
        location =    "Hong Kong",
      }
