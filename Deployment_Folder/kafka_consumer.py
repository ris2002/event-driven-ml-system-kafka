from kafka import KafkaConsumer
import json
import requests
FAST_API_URL='http://127.0.0.1:8000/dt_prediction'
consumer=KafkaConsumer('fake-jobs',bootstrap_servers='localhost:9092',auto_offset_reset='earliest',enable_auto_commit=True,group_id='fake-job-detector',value_deserializer=lambda x: json.loads(x.decode('utf-8')))
#consumer.subscribe(topics=['fake-jobs'])
for msg in consumer:
    print(f"Topic: {msg.topic}")
    

    row=msg.value
    response=requests.post(FAST_API_URL,json={"input":row})
    print('Prediction:',response.json())