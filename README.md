# python-deep-learning

Deep learning experiments with Keras 3 (TensorFlow backend) and NumPy.

## Requirements

- Python 3.11 (Homebrew: `brew install python@3.11`)
- The global pyenv Python (3.7) is too old for Keras 3 / TensorFlow — always use the project venv.

## Virtual environment

### First-time setup

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Everyday use

```sh
source .venv/bin/activate   # activate (prompt shows "(.venv)")
deactivate                  # leave the venv
```

## Adding new libraries

With the venv activated:

```sh
pip install <package>             # e.g. pip install matplotlib pandas
pip freeze > requirements.txt     # record exact versions
```

Commit the updated `requirements.txt` so the environment can be recreated.

Optional — Apple Silicon GPU acceleration:

```sh
pip install tensorflow-metal
```

## Running a script

With the venv activated:

```sh
python my_script.py
```

Or without activating:

```sh
.venv/bin/python my_script.py
```

Quick check that everything works:

```sh
python -c "import numpy, keras; print(numpy.__version__, keras.__version__, keras.backend.backend())"
```
