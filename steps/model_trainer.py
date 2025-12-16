from helper_trainer import Decision_Trees,KNN_classifier,XGBOOST_classifier
import logging
import pandas as pd
import numpy as np
from airflow.decorators import task



@task
def model_training()->None:
    try:
        pass
    except Exception as e :
        raise e
