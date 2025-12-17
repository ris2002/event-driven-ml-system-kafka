from data_ingestion import ingest_csv
from data_cleaner import data_cleaning
from airflow.decorators import task,dag
from model_trainer import model_training








@dag(dag_id='dag_train_pipeline', schedule='@once')
def training_pipeline():
    csv_path='/opt/airflow/Dataset_Folder/fake_job_postings.csv'
    task_ingestion=ingest_csv(csv_path)
    task_cleaning= data_cleaning()
    task_training=model_training()
    task_ingestion>>task_cleaning>>task_training

pipeline=training_pipeline()
