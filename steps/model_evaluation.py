from helper_evaluator import Acccuracy_Score_Class,Classification_Report_Class,Confusion_Matrix_Class
import mlflow
import pandas as pd
import numpy as np
import logging
from airflow.decorators import task
import pickle



@task
def model_evaluation_metrics()->None:
    try:
        mlflow.set_tracking_uri('http://host.docker.internal:5050')
        mlflow.set_experiment('fake_job_posting')

        with open('/opt/airflow/config/x_test.pkl','rb') as f:
            x_test=pickle.load(f)
        with open('/opt/airflow/config/y_test.pkl','rb') as f:
            y_test=pickle.load(f)

        with open('/opt/airflow/config/dt_model.pkl','rb') as f:
            dt_model=pickle.load(f)
        with open('/opt/airflow/config/knn_model.pkl','rb') as f:
            knn_model=pickle.load(f)
        dt_predicted_model=dt_model.predict(x_test)
        knn_model_predicted=knn_model.predict(x_test)
        
        accuracy_score_class=Acccuracy_Score_Class()
        confusion_matrix_class=Confusion_Matrix_Class()
        classification_report_class=Classification_Report_Class()

        accuracy_score_metrics_dt_model=accuracy_score_class.metrics(y_test,dt_predicted_model)
        accuracy_score_metrics_knn_model=accuracy_score_class.metrics(y_test,knn_model_predicted)
        True_Positive_dt,True_Negative_dt,False_Positive_dt,False_Negative_dt=confusion_matrix_class.metrics(y_test,dt_predicted_model)
        True_Positive_knn,True_Negative_knn,False_Positive_knn,False_Negative_knn=confusion_matrix_class.metrics(y_test,knn_model_predicted)
        dt_classification_report=classification_report_class.metrics(y_test,dt_predicted_model)
        knn_classification_report=classification_report_class.metrics(y_test,knn_model_predicted)


        with mlflow.start_run(run_name='dt_trained_model') as run:
            mlflow.log_metric('accuracy_scores', accuracy_score_metrics_dt_model)
            mlflow.log_metric('true_positive',True_Positive_dt)
            mlflow.log_metric('true_negative',True_Negative_dt)
            mlflow.log_metric('false_positive',False_Positive_dt)
            mlflow.log_metric('false_negative',False_Negative_dt)
            mlflow.log_metric('dt_macro_precision',
                      dt_classification_report['macro avg']['precision'])
            mlflow.log_metric('dt_macro_recall',
                      dt_classification_report['macro avg']['recall'])
            mlflow.log_metric('dt_macro_f1',
                      dt_classification_report['macro avg']['f1-score'])
            mlflow.log_metric('dt_weighted_f1',
                      dt_classification_report['weighted avg']['f1-score'])
        
        with mlflow.start_run(run_name='knn_trained_model') as run:
            mlflow.log_metric('accuracy_scores', accuracy_score_metrics_knn_model)
            mlflow.log_metric('true_positive',True_Positive_knn)
            mlflow.log_metric('true_negative',True_Negative_knn)
            mlflow.log_metric('false_positive',False_Positive_knn)
            mlflow.log_metric('false_negative',False_Negative_knn)
            mlflow.log_metric('knn_macro_precision',
                      knn_classification_report['macro avg']['precision'])
            mlflow.log_metric('knn_macro_recall',
                      knn_classification_report['macro avg']['recall'])
            mlflow.log_metric('knn_macro_f1',
                      knn_classification_report['macro avg']['f1-score'])
            mlflow.log_metric('knn_weighted_f1',
                      knn_classification_report['weighted avg']['f1-score'])


        


    except Exception as e :
        raise e 