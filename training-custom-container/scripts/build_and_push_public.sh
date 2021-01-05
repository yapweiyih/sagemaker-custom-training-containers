#!/bin/bash -x

display_usage() {
	echo "Usage: command ACCOUNT_ID REGION REPO_NAME"
}

if [ ! $# -eq 3 ]; then
    display_usage
    exit 1
fi
ACCOUNT_ID=$1
REGION=$2
REPO_NAME=$3
echo "ACCOUNT_ID: ${ACCOUNT_ID}"
echo "REPO_NAME: ${REPO_NAME}"
echo "REGION: ${REGION}"

# Build docker images
echo "***Building images***"
docker build -t $REPO_NAME ../docker
docker tag $REPO_NAME public.ecr.aws/i0z8o7s2/$REPO_NAME:latest

################################
# Public ecr - Region must be us-east-1
echo "***Create Repo***"
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/i0z8o7s2
aws ecr-public describe-repositories --region us-east-1 --repository-names $REPO_NAME || aws ecr-public create-repository --region us-east-1 --repository-name $REPO_NAME
docker push public.ecr.aws/i0z8o7s2/$REPO_NAME:latest
