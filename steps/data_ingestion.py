import logging
import pandas as pd
import pickle
from airflow.decorators import task

@task
def ingest_csv(csv_path)->pd.DataFrame:
    try:
        
        df=pd.read_csv(csv_path)
        with open('/opt/airflow/config/dataset.pkl','wb') as f:
            pickle.dump(df,f)
        logging.info('Successful in ingesting the csv file')


    except Exception as e:
        logging.error(f'Error Ingesting the csv file:{e}')
        raise e 
