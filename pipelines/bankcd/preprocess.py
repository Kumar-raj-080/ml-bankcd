import boto3, os, urllib.request
import numpy as np
import pandas as pd
from time import gmtime, strftime
import logging
logger = logging.getLogger('log')
logging.basicConfig(level=logging.INFO) # Set logging level and STDOUT handler

logger.info('bankcd:preprocess.py starts')

# prefix = 'sagemaker/DEMO-xgboost-dm'
prefix = "xgb"
my_region = 'us-east-1' #boto3.session.Session().region_name # set the region of the instance
bucket_name = 'tcb-bankcd' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET

s3 = boto3.resource('s3')

# try:
#     if  my_region == 'us-east-1':
#       s3.create_bucket(Bucket=bucket_name)
#     else: 
#       s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
#     logging.info('S3 bucket created successfully')
# except Exception as e:
#     logging.error('S3 error: ',e)

# Download the bank dataset as a csv file.
try:
  urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
  logger.info('Success: Data loaded into dataframe.')
except Exception as e:
  logger.error('Data load error: ',e)

# Read downloaded csv as a dataframe.
try:
    model_data = pd.read_csv('./bank_clean.csv',index_col=0)
    logger.info('Success: Data loaded into dataframe.')
except Exception as e:
    logger.error('Data load error ', e)

# Create train and test data sets and upload data sets to AWS directory
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
logger.info(train_data.shape, test_data.shape)
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv(f"{prefix}/train/train.csv", index=False, header=False)
pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv(f"{prefix}/test/test.csv", index=False, header=False)

logger.info('Success: pre-processed bank_clean.csv.')

# Upload training data into S3 bucket
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file(f"{prefix}/train/train.csv")
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file(f"{prefix}/test/test.csv")
logger.info('bankcd:preprocess.py ends')