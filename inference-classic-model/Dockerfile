# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.
FROM 520713654638.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-tensorflow-scriptmode:1.12-cpu-py3

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache --upgrade --use-feature=2020-resolver \
    flask gevent gunicorn \
    numpy==1.19.1 \
    pandas==1.0.5 \
    scikit-learn==0.23.2 \
    requests==2.24.0 \
    scipy==1.5.1 \
    catboost==0.24.1
    
    
# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY deployment_utility /opt/program
COPY predictor.py /opt/program
WORKDIR /opt/program

# Run the program that serves predictions
#ENTRYPOINT ["./serve"]

