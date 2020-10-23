# Processing steps

## Build container

./scripts/build_and_push.sh

- take note of container image names.
image_uri:
"342474125894.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-processor:latest"

## Input file and source code

Upload input file and source code to S3.
code:
s3://sagemaker-ap-southeast-1-342474125894/riotinto/preprocessing/code/processor.py

input:
"s3://sagemaker-ap-southeast-1-342474125894/riotinto/preprocessing/input"

output:
"s3://sagemaker-ap-southeast-1-342474125894/riotinto/preprocessing/output"

- Take note of input, output, code S3Uri

## Test run ScriptProcessor locally

cd notebook
python pipeline.py

## Deploy to cloudformation to be run by step function

cd template_folder/cloudformation
deploy stepfunction using rain

notes:

- statemachien is unable to take image name as input. Might be a bug.

## TODO

need to think about a way to pass sagemaker processing jobname to stepfunction. the name has to be unique
