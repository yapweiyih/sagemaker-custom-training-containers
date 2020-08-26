#!/bin/bash

ACCOUNT_ID=$1
REGION=$2
REPO_NAME=$3
SERVER="${ACCOUNT_ID}.dkr.ecr.ap-southeast-1.amazonaws.com"

# Login to retrieve base container if needed
# aws ecr get-login --no-include-email --registry-ids 763104351884 --region ${REGION} | awk '{print $6}' | docker login -u AWS --password-stdin 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com

# Dockerfile path is wrt execution work directory
docker build -f ../docker/Dockerfile -t $REPO_NAME ../docker

docker tag $REPO_NAME $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

# Login to ecr
# awscliv1
# aws ecr get-login --no-include-email --region $REGION --registry-ids $ACCOUNT_ID | awk '{print $6}' | docker login -u AWS --password-stdin $SERVER
# awscliv2
aws ecr get-login-password | docker login --username AWS --password-stdin ${SERVER}

aws ecr describe-repositories --repository-names $REPO_NAME || aws ecr create-repository --repository-name $REPO_NAME

docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

