# Reference

<https://docs.aws.amazon.com/sagemaker/latest/dg/build-container-to-train-script-get-started.html>

# Role

role = "arn:aws:iam::342474125894:role/service-role/AmazonSageMaker-ExecutionRole-20190405T234154"

# Build container

```
docker build -t pytorch-audio-classification-byoc -f Dockerfile .

```

# Test container local with mounting

```
docker run -v /home/ec2-user/efs_project/audio-classification-aws/data/:/opt/ml/data/ 839bddb53271 python /opt/ml/code/train.py --model m3

```

# Install docker-compose

```
sudo curl -L "https://github.com/docker/compose/releases/download/1.26.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

```

# Train locally with Sagemaker

```
from sagemaker.estimator import Estimator

role = "arn:aws:iam::342474125894:role/service-role/AmazonSageMaker-ExecutionRole-20190405T234154"

estimator = Estimator(image_name='pytorch-audio-classification-byoc',
   role=role,
   train_instance_count=1,
   train_instance_type='local')

estimator.fit()
```
