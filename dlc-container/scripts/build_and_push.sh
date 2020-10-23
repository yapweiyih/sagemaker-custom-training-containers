#!/bin/bash


#####################################################################
# Tested with AWSCLI v2
#
# AWSCLI v2 does not have get-login, use get-login-password instead
#
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
#####################################################################


ACCOUNT_ID='342474125894'
REGION='ap-southeast-1'
REPO_NAME='image-name'
SERVER="${ACCOUNT_ID}.dkr.ecr.ap-southeast-1.amazonaws.com"

# Login to retrieve base container if needed
# aws ecr get-login --no-include-email --registry-ids 763104351884 --region ${REGION} | awk '{print $6}' | docker login -u AWS --password-stdin 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com
aws ecr get-login-password | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com

# Dockerfile path is wrt execution work directory
docker build -f Dockerfile -t $REPO_NAME .

docker tag $REPO_NAME $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

# Login to ecr
# aws ecr get-login --no-include-email --region $REGION --registry-ids $ACCOUNT_ID | awk '{print $6}' | docker login -u AWS --password-stdin $SERVER
aws ecr get-login-password | docker login --username AWS --password-stdin ${SERVER}

aws ecr describe-repositories --repository-names $REPO_NAME || aws ecr create-repository --repository-name $REPO_NAME

docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

