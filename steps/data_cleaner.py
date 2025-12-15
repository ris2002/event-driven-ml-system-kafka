from airflow.decorators import task
from helper_cleaner import Pre_Proccessing_Strategies
import pickle
import logging


@task

def data_cleaning():
    
    try:
        pre_process=Pre_Proccessing_Strategies()

        with open('/opt/airflow/config/dataset.pkl','rb')as f:
            df=pickle.load(f)
            X_train,X_test,Y_train,Y_test=pre_process.final_stage(pre_process.intermidiate(pre_process.initial_stage(df)))
            pre_process.store_in_pkl(X_train,X_test,Y_train,Y_test)
    except Exception as e:
        raise e

