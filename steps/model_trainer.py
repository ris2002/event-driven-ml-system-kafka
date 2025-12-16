from helper_trainer import Decision_Trees,KNN_classifier,XGBOOST_classifier
import logging
import pandas as pd
import numpy as np
from airflow.decorators import task
import pickle
import mlflow
import mlflow.sklearn
import json



@task
def model_training()->None:
    try:
        config_path='/opt/airflow/config/config.json'
        with open(config_path,'r') as f:
            config=json.load(f)
        with open('/opt/airflow/config/x_train.pkl','rb') as f:
            x_train=pickle.load(f)
        with open('/opt/airflow/config/y_train.pkl','rb') as f:
            y_train=pickle.load(f)
        knn_model=KNN_classifier()
        decision_trees_classifier=Decision_Trees()
        xg_boost_classifier=XGBOOST_classifier()
        
        
        xgboost_trained_model=xg_boost_classifier.train_model(x_train,y_train,**config['xgb'])



        with mlflow.start_run(run_name='knn_trained_model') as run:
            knn_trained_model=knn_model.train_model(x_train,y_train,**config['knn'])
            logging.info('Successfully Trained KNN model')
            mlflow.sklearn.log_model(sk_model=knn_trained_model,artifact_path='knn_model')
            run_id=run.info.run_id
            mlflow.register_model(
            model_uri=f'runs:/{run_id}/knn',
            name='knn_trained_model' )
        with mlflow.start_run(run_name='Decision_Trees_trained_model') as run:
            decision_trees_model=decision_trees_classifier.train_model(x_train,y_train,**config['dt'])
            logging.info('Successfully Trained Decision Tree model')
            mlflow.sklearn.log_model(sk_model=decision_trees_model,artifact_path='decision_trees_model')
            run_id=run.info.run_id
            mlflow.register_model(
            model_uri=f'runs:/{run_id}/decision_tree_model',
            name='Decision_Trees_trained_model' )
        
        with mlflow.start_run(run_name='XGB_model') as run:
            xgboost_trained_model=xg_boost_classifier.train_model(x_train,y_train,**config['xgb'])
            logging.info('Successfully Trained XGB model')
            mlflow.sklearn.log_model(sk_model= xgboost_trained_model,artifact_path='xgboost_trained_model')
            run_id=run.info.run_id
            mlflow.register_model(
            model_uri=f'runs:/{run_id}/xgboost_trained_model',
            name='XGB_model' )

        
        with open('/opt/airflow/config/model_pkl_files/knn_model.pkl','wb') as f:
            pickle.dump(knn_trained_model,f)
        logging.info('Successfully put the knn_model in a pkl file')
        with open('/opt/airflow/config/model_pkl_files/dt_model.pkl','wb') as f:
            pickle.dump(decision_trees_model,f)
        logging.info('Successfully put the DT in a pkl file')

        with open('/opt/airflow/config/model_pkl_files/xgb_model.pkl','wb') as f:
            pickle.dump(xgboost_trained_model,f)
        logging.info('Successfully put the xgb_model in a pkl file')
        






        
    except Exception as e :
        logging.error(f'Error training the model:{e}')
        raise e
