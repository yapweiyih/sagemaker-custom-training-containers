#!/bin/bash

# Usage:
# ./predict.sh test.csv

# This is csv file location
payload=$1
content=${2:-text/csv}
echo "Inference input:"
echo ${payload}
echo

curl --data-binary @${payload} -H "Content-Type: ${content}" -v http://localhost:8080/invocations
