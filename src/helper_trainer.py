import logging
from abc import ABC,abstractmethod
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class Model_Training(ABC):
    @abstractmethod
    def train_model(self,x_test:np.ndarray,y_test:pd.DataFrame,**kwargs):
        pass


class Decision_Trees(Model_Training):
    def train_model(self,x_test:np.ndarray,y_test:pd.DataFrame,**kwargs): 
        try:
            dtc_model=DecisionTreeClassifier(**kwargs)
            training_model=dtc_model.fit(x_test,y_test)
            logging.info('Decision Tree model is succcesssfully trained')
            return training_model
        except Exception as e:
            logging.error(f'Error in training the deciasion tree classifier:{e}')
            raise e
class KNN_classifier(Model_Training):
    def train_model(self,x_test:np.ndarray,y_test:pd.DataFrame,**kwargs): 
        try:
            knn_model=KNeighborsClassifier(**kwargs)
            training_model=knn_model.fit(x_test,y_test)
            logging.info('KNN classifier model is succcesssfully trained')
            return training_model
        except Exception as e:
            logging.error(f'Error in training the KNN classifier classifier:{e}')
            raise e

class XGBOOST_classifier(Model_Training):
    def train_model(self,x_test:np.ndarray,y_test:pd.DataFrame,**kwargs): 
        try:
            #training model
            logging.info('XGBOOST_ classifier model is succcesssfully trained')
        except Exception as e:
            logging.error(f'Error in training the XGBOOST_ classifier classifier')