
#!/usr/local/bin/bash -x

# Pull the DLC base container 
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md

REGION='ap-southeast-1'
echo ${REGION}

#aws ecr get-login --no-include-email --registry-ids 763104351884 --region ${REGION} | awk '{print $6}' | docker login -u AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com
aws ecr get-login-password | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com

# Test pull
# docker pull 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.6.0-gpu-py36-cu101-ubuntu16.04

# debug container
#docker run -it -v/home/ec2-user/efs_project/sandbox/data:/mnt/data/ 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.6.0-cpu-py36-ubuntu16.04 bash
#docker run -it 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.6.0-cpu-py36-ubuntu16.04 bash
