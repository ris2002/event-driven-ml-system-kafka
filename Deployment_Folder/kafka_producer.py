from kafka import KafkaProducer
import logging
import numpy as np 
import pickle
import json
import time

def on_success(record_metadata):
    print(f"Success! Topic: {record_metadata.topic}, Partition: {record_metadata.partition}")

def on_error(excp):
    print(f"Error sending message: {excp}")

with open('/Users/rishilboddula/Desktop/MLOPS/Fake-JOB-POSTING-MLOPS/config/x_deploy.pkl','rb') as f:
    x_test=pickle.load(f)
producer=KafkaProducer(bootstrap_servers='localhost:9092',value_serializer=lambda v: json.dumps(v).encode('utf-8'))
topic='fake-jobs'
for row in x_test:
    tx_row = row.toarray().flatten().tolist()

    print('sent_data')

    #Kafka doesnt understad what an nd array iss so we convert it to lidt anssd send it 

    producer.send(topic,value=tx_row).add_callback(on_success).add_errback(on_error)
    producer.flush()
    time.sleep(10)
print('producer closed')
producer.close()