import logging
from abc import ABC,abstractmethod
import numpy as np
import pandas as pd


class Model_Training(ABC):
    @abstractmethod
    def train_model(self,x_test:np.ndarray,y_test:pd.DataFrame,**kwargs):
        pass


class Decision_Trees(Model_Training):
    def train_model(self,x_test:np.ndarray,y_test:pd.DataFrame,**kwargs): 
        try:
            #training model
            logging.info('Decision Tree model is succcesssfully trained')
        except Exception as e:
            logging.error(f'Error in training the deciasion tree classifier')
class KNN_classifier(Model_Training):
    def train_model(self,x_test:np.ndarray,y_test:pd.DataFrame,**kwargs): 
        try:
            #training model
            logging.info('KNN classifier model is succcesssfully trained')
        except Exception as e:
            logging.error(f'Error in training the KNN classifier classifier')

class XGBOOST_classifier(Model_Training):
    def train_model(self,x_test:np.ndarray,y_test:pd.DataFrame,**kwargs): 
        try:
            #training model
            logging.info('XGBOOST_ classifier model is succcesssfully trained')
        except Exception as e:
            logging.error(f'Error in training the XGBOOST_ classifier classifier')