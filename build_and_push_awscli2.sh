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

# Login to retrieve base container if needed
# awscliv1 (deprecated)
# aws ecr get-login --no-include-email --registry-ids 763104351884 --region ${REGION} | awk '{print $6}' | docker login -u AWS --password-stdin 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com

# Authenticate docker to access specific account id/ecr server
aws ecr get-login-password | docker login --username AWS --password-stdin 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com

# Build docker images
echo "***Building images***"
docker build -t $REPO_NAME ../docker
docker tag $REPO_NAME $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

# Authenticate docker to ecr
# awscliv1 (deprecated)
# aws ecr get-login --no-include-email --region $REGION --registry-ids $ACCOUNT_ID | awk '{print $6}' | docker login -u AWS --password-stdin $SERVER
# awscliv2
SERVER="${ACCOUNT_ID}.dkr.ecr.ap-southeast-1.amazonaws.com"
aws ecr get-login-password | docker login --username AWS --password-stdin ${SERVER}

# Create ecr repo is not exist
echo "***Create Repo***"
aws ecr describe-repositories --repository-names $REPO_NAME || aws ecr create-repository --repository-name $REPO_NAME

# Push image to ecr
echo "***Push to Repo***"
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

