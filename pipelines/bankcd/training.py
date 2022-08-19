# import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime, strftime
from sagemaker.predictor import csv_serializer
import logging
logger = logging.getLogger('log')
logging.basicConfig(level=logging.INFO) # Set logging level and STDOUT handler

# Define IAM Role
role = get_execution_role()

prefix = 'xgb'
my_region = boto3.session.Session().region_name
bucket_name = 'tcb-bankcd'


# try:
#     model_data = pd.read_csv('./bank_clean.csv',index_col=0)
#     print('Success: Data loaded into dataframe.')
# except Exception as e:
#     print('Data load error: ',e)
try:
    # this line automatically looks for the XGBoost image URI and builds an XGBoost container.
    xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")
    logger.info("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + xgboost_container + " container for your SageMaker endpoint.")
except Exception as e:
    logger.error('Failure in creating xgboost container ', e)
try:
    train_input = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/{}'.format(bucket_name, prefix,"train/train.csv"), content_type='csv')
    validation_input = sagemaker.inputs.TrainingInput("s3://{}/{}/{}".format(bucket_name, prefix, "test/test.csv"), content_type="csv")
    sess = sagemaker.Session()
    xgb_model = sagemaker.estimator.Estimator(xgboost_container,role, instance_count=1, instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket_name, prefix),sagemaker_session=sess)
    xgb_model.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)
    # Fit model on training data and deploy model
    xgb_model.fit({'train': train_input, "validation": validation_input}, wait=True)
    logger.info('TCB XGboost demo model training completed', xgb_model.model_data)
except Exception as e:
    logger.error('Failure in train model ', e)
#xgb_predictor = xgb.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
# Save model as a pickle file to be used in evaluation.py
# filename = 'xgb_pred.pkl'
# pickle.dump(xgb, open(filename, 'wb'))


