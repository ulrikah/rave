# rave

## Setup

### Non-python dependencies

The project relies on Csound for rendering audio. Csound be downloaded from the [homepage](https://csound.com/index.html).

### Project setup

The project is set up with [poetry](https://github.com/python-poetry/poetry). The easiest way to install the dependencies for this project is to install `poetry`, and then execute the following commands inside the root folder of this project (where the `pyproject.toml` is located):

```bash
poetry config virtualenvs.in-project true # optional
poetry shell # initialize the virtual environment
poetry install
poetry run task test # run tests to verify that things work properly
```

## Launching a training process

A training process is launched by running `python rave/train.py`.

By default, the training script uses the default configuration located in `rave/configs/default.toml`. This is where settings are defined, such as which feature extractors to use.

The training process renders audio files at the end of every episode so that it is possible to hear the results as the training goes on. These audio files are rendered to `rave/bounces`.

## Testing a trained model

By default, checkpoints are saved every 100 iterations in `rave/ray_results/<name-of-experiment>` during training. You can use any of these checkpoints to run inference on a trained model by running `rave/inference.py` with the `--checkpoint` flag. Make sure to use the full path to the checkpoint, since the program is not very forgiving with path typos yet. Example:

`python rave/inference.py --checkpoint rave/ray_results/SAC_default_2021-03-02_22-52-02/SAC_CrossAdaptiveEnv_8566d_00000_0_2021-03-02_22-52-02/checkpoint_500/checkpoint-500`

It is also possible to use other sounds than the default ones when testing out the model. Check the possible flags you can set by running with the `--help` flag.

## Inspecting results from the training process

All information regarding the training process is saved to `rave/ray_results/`. This information is saved in a [TensorBoard](https://www.tensorflow.org/tensorboard)-compliant format, which enables using TensorBoard to inspect the results. To do so, simply point TensorBoard to the to appropriate folder by typing this in a new terminal window:

```bash
tensorboard --logdir rave/ray_results
```

Note that the `tensorboard` command is installed with the dependencies of this project. As such, you need to activate the virtual environment beforehand with `poetry shell` (unless you already have `tensorboard` installed globally).
