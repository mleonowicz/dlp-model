## Info

This repository contains code for a simple, end-to-end ML workflow for traffic sign classification for [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset. The workflow is implemented with [Amazon SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html). Pipeline is defined with [SageMaker SDK](https://sagemaker.readthedocs.io/en/stable/).


The repository is based on a template described in [SageMaker MLOps Project Walkthrough](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects-walkthrough.html). Additional references follow in:
1. https://github.com/aws-samples/aws-sagemaker-pipelines-skin-classification/
2. https://github.com/aws-samples/amazon-sagemaker-pipelines-mxnet-image-classification/

The pipeline:
  * Splits the dataset and converts it to RecordIO format in  ([pipelines/road-sign/pipeline.py#L112](https://github.com/mleonowicz/dlp-model/blob/main/pipelines/road-sign/pipeline.py#L112)). It is assumed, that the dataset is uploaded to S3 beforehand.
  * Trains a classification model with transfer learning based on an already trained model ([pipelines/road-sign/pipeline.py#L151](https://github.com/mleonowicz/dlp-model/blob/main/pipelines/road-sign/pipeline.py#L151)).
  * Evaluates the model on the test dataset ([pipelines/road-sign/pipeline.py#L199](https://github.com/mleonowicz/dlp-model/blob/main/pipelines/road-sign/pipeline.py#L199)).
  * Registers the model to a model registry ([pipelines/road-sign/pipeline.py#L251](https://github.com/mleonowicz/dlp-model/blob/main/pipelines/road-sign/pipeline.py#L251)) when mean square error from the evaluation is lower than a specified threshold ([pipelines/road-sign/pipeline.py#L273](https://github.com/mleonowicz/dlp-model/blob/main/pipelines/road-sign/pipeline.py#L273)).

Model deployment code will be provided in another [dlp-deploy repository](https://github.com/mleonowicz/dlp-deploy).

Team members:
* [@kjpolak](https://github.com/kjpolak/)
* [@mleonowicz](https://github.com/mleonowicz/)
* [@madziejm](https://github.com/madziejm/)


## Code layout 

```
|-- codebuild-buildspec.yml
|-- CONTRIBUTING.md
|-- pipelines
|   |-- road-sign
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   `-- preprocess.py
|   |-- create_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
|-- README.md
|-- sagemaker-pipelines-project.ipynb
|-- setup.cfg
|-- setup.py
|-- tests
|   `-- test_pipelines.py
`-- tox.ini
```

## Code description

A description of some of the artifacts is provided below:
<br/><br/>
Your codebuild execution instructions. This file contains the instructions needed to kick off an execution of the SageMaker Pipeline in the CICD system (via CodePipeline). You will see that this file has the fields definined for naming the Pipeline, ModelPackageGroup etc. You can customize them as required.

```
|-- codebuild-buildspec.yml
```

<br/><br/>
The pipeline artifacts, which includes a pipeline module defining the required `create_pipeline` method that returns an instance of a SageMaker pipeline, a preprocessing script that is used in feature engineering, and a model evaluation script to measure the Mean Squared Error of the model that's trained by the pipeline. This is the core business logic, and if you want to create your own folder, you can do so, and implement the `create_pipeline` interface as illustrated here.

```
|-- pipelines
|   |-- abalone
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   `-- preprocess.py

```
<br/><br/>
Utility modules for getting pipeline definition jsons and running pipelines (you do not typically need to modify these):

```
|-- pipelines
|   |-- create_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
```
<br/><br/>
Python package artifacts:
```
|-- setup.cfg
|-- setup.py
```
<br/><br/>
A stubbed testing module for testing your pipeline as you develop:
```
|-- tests
|   `-- test_pipelines.py
```
<br/><br/>
The `tox` testing framework configuration:
```
`-- tox.ini
```

## Running

Manual deployment from a local host is possible with

``` bash
pip install . && python pipelines/run_pipeline.py --module-name road_sign.pipeline --role-arn <provide role here> --kwargs '{"region": "<provide region here>"}'
```

