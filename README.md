# Amazon SageMaker custom training containers

Implementations of Amazon SageMaker-compatible custom containers for training.

# XGB container

- <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone_dist_script_mode.ipynb>
- <https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html>
- sagemaker-container and sagemaker-inference have been installed, refer to requirements.txt file in Dockerfile
- <https://github.com/aws/sagemaker-xgboost-container/blob/master/docker/1.0-1/final/Dockerfile.cpu>

# PyTorch container

- <https://github.com/aws/sagemaker-pytorch-training-toolkit>



# Helper container

scp -i ~/Documents/awskey/sg-aws-sandbox-keypair.pem  ec2-user@ec2a2:/home/ec2-user/efs_project/audio-classification-aws/notebooks/build_and_push.sh  notebook/
