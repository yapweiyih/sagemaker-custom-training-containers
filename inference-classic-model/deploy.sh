#!/usr/bin/env bash

# Script shows how to build the Docker image, it to ECR, create a SageMaker
# model, endpoint configuration, and endpoint.
#
# ARGUMENTS:
#
#   image:
#      This will be used as the image on the local machine and combined
#      with the account and region to form the repository name for ECR.
#      This is also used to name the model name and endpoint name.
#
#  [s3_model_location]:
#      This is an optional argument for where the model will be stored in s3.
#      It defaults to s3://sdu-machine-learning/${image_name}/input/model.tar.gz
#
#  [sagemaker_execution_role]:
#      This is an optional argument for which role SageMaker will use to create
#      the model object, the endpoint configuration, and the enpoint itself.
#      It defaults to AmazonSageMaker-ExecutionRole-20180306T104914
#
# EXAMPLE USAGE:
# ./deploy.sh iris-model
# ./deploy.sh iris-model s3://sagemaker-demo-samples/scikit-learn-deploy/trained-model/model.tar.gz AmazonSageMaker-ExecutionRole-20171204T150334

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1
s3_model_location=${2:-s3://wy-sandbox-singapore/sagemaker/model/${image}/input/model.tar.gz}
sagemaker_service_role=${3:-AmazonSageMaker-ExecutionRole-20190405T234154}
user=default
endpoint_prefix='rtc360'


# Check that the image name is provided
if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name> [<s3_model_location>] [<sagemaker_service_role>]"
    exit 1
fi

# STEP 1: UPLOAD MODEL TO S3

# Tar and zip the model
tar czvf model.tar.gz model.pkl

# Upload the model to S3
aws s3 mv model.tar.gz ${s3_model_location} --profile ${user}

chmod +x deployment_utility/serve

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text --profile ${user})
echo ${account}

if [ $? -ne 0 ]
then
    exit 255
fi

######################################################################
# STEP 2: CREATE THE DOCKER CONTAINER
######################################################################

# Get the region defined in the current configuration (default to us-east-1 if none defined)
region=$(aws configure get region)
region=${region:-ap-southeast-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${image}" --profile ${user} > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" --profile ${user} > /dev/null
fi

# Get the login command from ECR and execute it directly
# $(aws ecr get-login --region ${region} --no-include-email --profile ${user} --registry-ids 520713654638)
SERVER="${account}.dkr.ecr.ap-southeast-1.amazonaws.com"
aws ecr get-login-password | docker login --username AWS --password-stdin ${SERVER}
aws ecr get-login-password | docker login --username AWS --password-stdin "520713654638.dkr.ecr.ap-southeast-1.amazonaws.com"

# Build the docker image locally with the image name.
docker build -t ${image} .

echo
echo Docker image built ...

docker tag ${image} ${fullname}


# Push image to ECR with the full name
docker push ${fullname}

echo
echo Docker image pushed ...

exit
######################################################################
# STEP 3: CREATE THE ENDPOINT
######################################################################

endpoint_name=${endpoint_prefix}'-'$(date '+%Y%m%d-%H%M%S')

# Create the model object
aws sagemaker create-model \
    --model-name ${endpoint_name} \
    --primary-container \
        Image=${fullname},ModelDataUrl=${s3_model_location} \
    --execution-role-arn arn:aws:iam::${account}:role/service-role/${sagemaker_service_role} \
    --profile ${user}

# Create endpoint configuration
aws sagemaker create-endpoint-config \
    --endpoint-config-name ${endpoint_name} \
    --production-variants \
        VariantName=dev,ModelName=${endpoint_name},InitialInstanceCount=1,InstanceType=ml.m4.xlarge,InitialVariantWeight=1.0 \
    --profile ${user}

# Create the actual endpoint
aws sagemaker create-endpoint \
    --endpoint-name ${endpoint_name} \
    --endpoint-config-name ${endpoint_name} \
    --profile ${user}

echo Creating endpoint ${endpoint_name} ...
echo
echo To check on the status use:
echo "   " !aws sagemaker list-endpoints --name-contains ${endpoint_name}
