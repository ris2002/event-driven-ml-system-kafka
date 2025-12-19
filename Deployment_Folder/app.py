import logging
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from abc import ABC, abstractmethod
import numpy as np
app=FastAPI()
mlflow.set_tracking_uri('http://host.docker.internal:5050')



class Input_Text(BaseModel):
    input:List[float]

@app.post('/knn_prediction')
def predict_using_knn(input_row:Input_Text):
        try:
            loaded_knn_model=mlflow.pyfunc.load_model('models:/Production_KNN_model/1')
            np_input=np.array(input_row.input).reshape(1, -1)
            predicting_result=loaded_knn_model.predict(np_input)
            logging.info('Prediction Done')
            print('prediction',predicting_result.tolist())

            return{"prediction": predicting_result.tolist()}
        except Exception as e:
            logging.error(f'ERR Predicting using KNN model:{e}')
@app.post('/dt_prediction')
def predict_using_dt(input_row:Input_Text):
        try:
            loaded_decision_tree_model=mlflow.pyfunc.load_model('models:/Production_Decision_Tree_Model/1')
            np_input=np.array(input_row.input).reshape(1, -1)
            predicting_result=loaded_decision_tree_model.predict(np_input)
            logging.info('Prediction Done')
            print('prediction',predicting_result.tolist())
            return{"prediction": predicting_result.tolist()}
        except Exception as e:
            logging.error(f'ERR Predicting using DT model:{e}')


