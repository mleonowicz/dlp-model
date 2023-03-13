"""Workflow pipeline script for road-sign pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements create_pipeline(**kwargs) method.
"""

from pathlib import Path

import boto3
import sagemaker
import sagemaker.inputs
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.mxnet import MXNetProcessor

from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from pipelines.road_sign.constants import TRAIN_NIMAGES

BASE_DIR = Path(__file__).resolve().parent


def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session

    Returns:
        botocore.client.SageMaker instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def create_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def create_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def create_preprocessing_step(
    input_archive: ParameterString,
    instance_type,
    base_job_name,
    sagemaker_session,
    role,
):
    processor = MXNetProcessor(
        framework_version="1.8.0",
        py_version="py37",
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        base_job_name=base_job_name,
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_args = processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train/"),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation/",
            ),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test/"),
        ],
        source_dir=str(BASE_DIR),  # copy all the module
        code=str("preprocess.py"),
        arguments=["--input-data", input_archive],
    )
    return ProcessingStep(
        name="PreprocessRoadSignData",
        step_args=step_args,
        cache_config=CacheConfig(
            enable_caching=True, expire_after="p30d"
        ),  # expire after 30 days
    )


def create_training_step(
    preprocessing_step, image_uri, instance_type, role, output_path
):
    inputs = {
        "train": sagemaker.inputs.TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="application/x-recordio",
            s3_data_type="S3Prefix",
            input_mode="Pipe",
        ),
        "validation": sagemaker.inputs.TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "validation"
            ].S3Output.S3Uri,
            content_type="application/x-recordio",
            s3_data_type="S3Prefix",
            input_mode="Pipe",
        ),
    }
    # reference: https://docs.aws.amazon.com/sagemaker/latest/dg/IC-Hyperparameter.html
    hyperparameters = {
        "num_layers": 18,
        "use_pretrained_model": 1,
        # "augmentation type": "crop_color_transform", # Weird, this gives unexpected "ClientError: Additional hyperparameters are not allowed ('augmentation type' was unexpected)"". This field should be accepted according to the schema. Don't augment for now.
        "num_classes": 43,
        "num_training_samples": TRAIN_NIMAGES,
        "epochs": 1,
        "learning_rate": 1e-5,
    }
    classifier = Estimator(
        hyperparameters=hyperparameters,
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        volume_size=30,
        max_run=9 * 60 * 60,  # let's say we can afford 9 hours
        output_path=output_path,
    )
    return (
        TrainingStep(name="TrainModel", estimator=classifier, inputs=inputs),
        classifier,
    )


def create_evaluation_step(
    preprocessing_step,
    training_step,
    base_job_name,
    role,
    pipeline_session,
):
    processor = MXNetProcessor(
        framework_version="1.8.0",
        py_version="py37",
        command=["python3"],
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name=base_job_name,
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = processor.run(
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test/",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", source="/opt/ml/processing/evaluation"
            ),
        ],
        code=str(BASE_DIR / "evaluate.py"),
    )
    evaluation_report = PropertyFile(
        name="RoadSignEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    return (
        ProcessingStep(
            name="EvaluateRoadSignModel",
            step_args=step_args,
            property_files=[evaluation_report],
        ),
        evaluation_report,
    )


def create_model_registration_step(training_step, estimator, group_name):
    return RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=[
            "image/jpeg",
            "application/x-recordio",
            "image/png",
            "image/jpeg",
            "application/x-image",
        ],
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge"],
        transform_instances=[
            "ml.m5.xlarge"
        ],  # I don't care about BatchTransform yet SDK requires an instance here
        model_package_group_name=group_name,
        approval_status="PendingManualApproval",
    )


def create_mse_cond_registration_step(
    register_step, evaluation_step, evaluation_report: PropertyFile
):
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="classification_metrics.accuracy.value",
        ),
        right=0.6,
    )
    return ConditionStep(
        name="MSECond", conditions=[cond_lte], if_steps=[register_step], else_steps=[]
    )


def create_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="RoadSignPackageGroup",
    pipeline_name="RoadSignPipeline",
    base_job_prefix="RoadSign",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.p2.xlarge",  # supress lovely "Instance type X is not supported by algorithm image-classification; only GPU instances are supported." limitation
    sagemaker_project_arn=None,  # don't remove without refactoring codebuild-buildspec.yml and _utils.py where it's fixed
):
    """Gets a SageMaker ML Pipeline instance working with road sign data.

    Args:
        region: AWS region name to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = create_pipeline_session(region, default_bucket)

    input_archive = ParameterString(
        name="GtsrbS3Uri",  # todo document it
        default_value=f"s3://sagemaker-{region}-830437619288-datasets/GTSRB.zip",
    )

    image_classification_image_uri = sagemaker.image_uris.retrieve(
        "image-classification", region
    )  # actually MXnet Docker image

    preprocessing_step = create_preprocessing_step(
        input_archive=input_archive,
        instance_type=processing_instance_type,
        base_job_name=f"{base_job_prefix}/road-sign-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )
    training_step, classifier = create_training_step(
        preprocessing_step=preprocessing_step,
        image_uri=image_classification_image_uri,
        instance_type=training_instance_type,
        role=role,
        output_path=f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/RoadSignTrain",
    )

    evaluation_step, evaluation_report = create_evaluation_step(
        preprocessing_step,
        training_step,
        f"{base_job_prefix}/script-road-sign-eval",
        role,
        pipeline_session,
    )

    registration_step = create_model_registration_step(
        training_step, classifier, model_package_group_name
    )

    mse_cond_registration_step = create_mse_cond_registration_step(
        registration_step, evaluation_step, evaluation_report
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[
            input_archive,
            processing_instance_type,
            training_instance_type,
        ],
        steps=[
            preprocessing_step,
            training_step,
            evaluation_step,
            mse_cond_registration_step,
        ],
    )
