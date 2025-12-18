from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import logging
from sklearn.base import ClassifierMixin
import pandas as pd
import numpy as np
from abc import ABC,abstractmethod
from typing import Tuple
from typing_extensions import Annotated


class Evaluating_Strategies(ABC):
    @abstractmethod
    def metrics(self,y_test:pd.DataFrame,predicted_model:np.ndarray):
        pass

class Acccuracy_Score_Class(Evaluating_Strategies):
    def metrics(self,y_test:pd.DataFrame,predicted_model:np.ndarray)->float:
        try:
            accuracy_score_metric=accuracy_score(y_test,predicted_model)
            logging.info('Calculated accuracy_score')
            return accuracy_score_metric
        except Exception as e:
            logging.error(f'failed to calculate accuracy_score:{e}')
            raise e 
class Confusion_Matrix_Class(Evaluating_Strategies):
    def metrics(self,y_test:pd.DataFrame,predicted_model:np.ndarray)->Tuple[Annotated[int,'True_Positive'],Annotated[int,'True_Negative'],Annotated[int,'False_Positive'],Annotated[int,'False_Negative']]:
        try:
            matrix=confusion_matrix(y_test,predicted_model)
            True_Negative=int(matrix[0][0])
            True_Positive=int(matrix[1][1])
            False_Negative=int(matrix[1][0])
            False_Positive=int(matrix[0][1])

            logging.info('Calculated confusion_matrix')

            return True_Positive,True_Negative,False_Positive,False_Negative
            
        except Exception as e:
            logging.error(f'failed to calculate confusion_matrix:{e}')

            raise e
        
class Classification_Report_Class(Evaluating_Strategies):
    def metrics(self,y_test:pd.DataFrame,predicted_model:np.ndarray)->dict:
        try:
            reports=classification_report(y_test,predicted_model,output_dict=True)
            logging.info('Calculated classification report')
            return reports
            

        except Exception as e:
            logging.error(f'failed to calculate classification_report:{e}')

            raise e