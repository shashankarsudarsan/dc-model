import pickle
import boto3
import tempfile
import os
import ctypes
import uuid
import sklearn
from sklearn import preprocessing
import numpy as np



for d, dirs, files in os.walk('lib'):
    for f in files:
        if f.endswith('.a'):
            continue
        ctypes.cdll.LoadLibrary(os.path.join(d, f))

s3 = boto3.resource('s3')

def classify_document(document):
    docu_file = open(document, 'r')
    text = [docu_file.readline()]
    bucket = 'dc-model-bucket'
    key = 'dc_model.pkl'
    with open('/tmp/model.pkl', 'wb') as data:
        s3.Bucket(bucket).download_fileobj(key, data)
    with open('/tmp/model.pkl','rb') as f:
        [logreg, tfidf, cv] = pickle.load(f)
    ans = logreg.predict(tfidf.transform((cv.transform(text))))	
    labels = ['APPLICATION', 'BILL', 'BILL BINDER', 'BINDER', 'CANCELLATION NOTICE', 'CHANGE ENDORSEMENT', 'DECLARATION', 'DELETION OF INTEREST', 'EXPIRATION NOTICE','INTENT TO CANCEL NOTICE', 'NON-RENEWAL NOTICE', 'POLICY CHANGE', 'REINSTATEMENT NOTICE', 'RETURNED CHECK']
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    final_ans = le.inverse_transform(ans)
    return final_ans

def lambda_handler(event, context):
    # TODO implement
	results = []
	for record in event['Records']:
		bucket = record['s3']['bucket']['name']
		key = record['s3']['object']['key']

	print('Running Document Classification ...')
	print('Document to be processed, from: bucket [%s], object key: [%s]' % (bucket, key))

	# load image
	tmp = tempfile.NamedTemporaryFile()
	with open(tmp.name, 'wb') as f:
	  s3.Bucket(bucket).download_file(key, tmp.name)
	  tmp.flush()
	  prediction_label = classify_document(tmp.name)
	  results.append('(Answer - %s)' % (prediction_label))

	print(results)
	return results    