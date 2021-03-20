# Getting Started

This is a sample package showing how a python model would be setup to run on
aibakery.

The model is can be found in `notebooks/<model-name>.joblib`

Models can be imported from `from example_model.model.<model name> import predict`

`<model name>` available:
- `random_model`
- `mnist_svc_model`

> #### Note
> We will need to override the default model path to `./notebooks`. See how to run.


## How to run
### If you have cloned the repo
- Install the requirements
- Run
    ```
    MODEL_LOCATION=./notebooks python -m example_model.main_service
    ```
    > What does this do?
    >
    > We are setting the model location using the environment variable `MODEL_LOCATION`

### If you have installed the package
- Run
  ```
  python -m example_model.main_service
  ```

## Requirements
### Running locally
All dependencies are listed in the requirements.txt file.

### Packaging
Ensure the `wheel` package is installed

## How to
### Building .whl/tar.gz
To build the package. From the root directory of the project run:
```shell
python3 setup.py sdist bdist_wheel
```
This will generate a .whl and .tar.gz in the dist directory.
