import pandas as pd

from features import data_agregator
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import xgboost as xgb


class DelayModel:

    def __init__(
        self
    ):
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)


    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data = data_agregator(data)
        training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state = 111)
        features = pd.concat([
                                pd.get_dummies(training_data['OPERA'], prefix = 'OPERA'),
                                pd.get_dummies(training_data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
                                pd.get_dummies(training_data['MES'], prefix = 'MES')], 
                                axis = 1
                                )
        if target_column is not None:
            return features, training_data[target_column]
        else:
            return features



    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        
        x_train, _ , y_train, _ = train_test_split(features, target, test_size = 0.33, random_state = 42)
        self._model.fit(x_train, y_train)


    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """

        predictions = self._model.predict(features)
        predictions = [1 if y_pred > 0.5 else 0 for y_pred in predictions]

        return predictions