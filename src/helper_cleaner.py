import logging
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.compose import ColumnTransformer




nltk.download('stopwords')
stop_words=set(stopwords.words('english'))
stemmer=PorterStemmer()
class Pre_Proccessing_Strategies:
    def initial_stage(self,df:pd.DataFrame)->pd.DataFrame:
        try:
            df=df.drop(columns=['salary_range', 'department','title'])
            text_cols=['description','benefits','requirements','company_profile']
            for col in text_cols:
                df[col]=df[col].str.lower()
            df['location']=df['location'].fillna('GB, LND, London')
            cols = ['required_experience','required_education','function','industry','employment_type','description','benefits', 'requirements','company_profile']       
            df[cols] = df[cols].fillna('Unknown')
            logging.info('Basic pre-processinng like droping the cols, filling the na values are done ')
            return df


        except Exception as e:
            logging.error(f'Error at cleaning the data in the INITIAL STAGE:{e}')
            raise e
    def intermidiate(self,df:pd.DataFrame)->pd.DataFrame:

        try:
            def more_preprocess(text):
                words=text.split()
                words=[stemmer.stem(w) for w in words if  w not in stop_words]# TFID vecctorizoer expects a str not a list
                return ' '.join(words)
            text_cols=['description','benefits','requirements','company_profile']
            for col in text_cols:
                df[col]=df[col].apply(more_preprocess)

            data=df
            logging.info(f'Done Stemming')
            return data

        except Exception as e:
            logging.error(f'Error in Stemming :{e}')
            raise e
    
    def final_stage(self,df:pd.DataFrame)->Tuple[Annotated[np.ndarray,"x_Train"],Annotated[np.ndarray,"x_Test"],Annotated[pd.DataFrame,"y_Train"],Annotated[pd.DataFrame,"y_Test"]]:
        try:
            X=df.drop(columns='fraudulent')
            Y=df['fraudulent']
            x_train_df,x_test_valid_df,y_train,y_test_valid=train_test_split(X,Y,test_size=0.35,random_state=42)
            x_test_df,x_deploy_df,y_test,y_deploy=train_test_split(x_test_valid_df,y_test_valid,test_size=0.5,random_state=45)
            text_cols=['description','benefits','requirements','company_profile']
            x_train_df['all_text'] = x_train_df[text_cols].apply(lambda row: ' '.join(row), axis=1) # new col
            x_test_df['all_text']  = x_test_df[text_cols].apply(lambda row: ' '.join(row), axis=1)
            x_deploy_df['all_text']  = x_deploy_df[text_cols].apply(lambda row: ' '.join(row), axis=1)
            cat_cols=['location','function','industry','required_education','required_experience','employment_type']
            binary_cols = ['has_company_logo', 'has_questions', 'telecommuting']
            '''Concatenates all text for a job posting into a single string
               Now TF-IDF sees the entire job posting as one document
               Helps the model capture relationships between words in different columns'''
            preprocessor=ColumnTransformer(
                transformers=[
                    ('tfid_vectorizer',TfidfVectorizer(max_features=5000,ngram_range=(1,2)),'all_text'),
                    ('one_hot_encoding',OneHotEncoder(handle_unknown='ignore'), cat_cols)
                ],
                remainder='drop')
            x_train=preprocessor.fit_transform(x_train_df)
            x_test=preprocessor.transform(x_test_df)
            x_deploy=preprocessor.transform(x_deploy_df)
            with open('/opt/airflow/config/x_deploy.pkl','wb') as f:
                pickle.dump(x_deploy,f)
            with open('/opt/airflow/config/y_deploy.pkl','wb') as f:
                pickle.dump(y_deploy,f)
            logging.info('Sucessfully  Vectorizing or spliting data')
            return x_train,x_test,y_train,y_test

            




            
        except Exception as e:
            logging.error(f'Error in Vectorizing or spliting data:{e}')
            raise e
    
    def store_in_pkl(
    self,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame
) -> None:
        
        try:
            with open('/opt/airflow/config/x_train.pkl', 'wb') as f:
                pickle.dump(x_train, f)
            with open('/opt/airflow/config/y_train.pkl', 'wb') as f:
                pickle.dump(y_train, f)
            with open('/opt/airflow/config/x_test.pkl', 'wb') as f:
                pickle.dump(x_test, f)
            with open('/opt/airflow/config/y_test.pkl', 'wb') as f:
                pickle.dump(y_test, f)
            logging.info('Successfully stored train/test datasets in pickle files.')
        except Exception as e:
            raise e




        
        
