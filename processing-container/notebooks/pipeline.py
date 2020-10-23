"""
SageMaker processing job

Key concepts:
- By default, the code "processor.py" will be placed under "/opt/ml/processing/input/code"
- recommend to put input files under "/opt/ml/processing/input/data", so that we can easily read all the files without filtering
- recommend to put output files under "/opt/ml/processing/output"
- data science step function sdk can be used to join all together with subsequent training job
"""

from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

image_uri = "342474125894.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-processor:latest"
role = "arn:aws:iam::342474125894:role/service-role/AmazonSageMaker-ExecutionRole-20190405T234154"

script_processor = ScriptProcessor(
    role=role,
    image_uri=image_uri,
    command=["python3"],
    instance_count=1,
    instance_type="ml.m5.xlarge",
)


script_processor.run(
    code="s3://sagemaker-ap-southeast-1-342474125894/riotinto/preprocessing/code/processor.py",
    inputs=[
        ProcessingInput(
            # Include data files, Can be s3 or local path
            source="s3://sagemaker-ap-southeast-1-342474125894/riotinto/preprocessing/input",
            destination="/opt/ml/processing/input/data",
            input_name="parquet",
            s3_data_type="S3Prefix",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination="s3://sagemaker-ap-southeast-1-342474125894/riotinto/preprocessing/output",
            output_name="output",
        ),
    ],
    # Must be list of str
    arguments=["--option", "1"],
    wait=True,
)
