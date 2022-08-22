FROM python:3.8

RUN pip3 install --no-cache scikit-learn pandas joblib flask requests boto3 tabulate

COPY pipelines/bankcd/preprocess.py /usr/bin/preprocess
COPY pipelines/bankcd/training.py /usr/bin/train
COPY pipelines/bankcd/evaluate.py /usr/bin/serve

RUN chmod 755 /usr/bin/train /usr/bin/serve

EXPOSE 8080
 
