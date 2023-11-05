from pathlib import Path
import pandas as pd

from challenge.features import data_agregator
# from features import data_agregator
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import xgboost as xgb


class DelayModel:
    
    MODEL_ROOT_PATH = str(Path(__file__).parent / "model.json")

    top_10_features = [
    "OPERA_Latin American Wings", 
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air"
    ]

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
            # Returning target as a DataFrame
            target = training_data[target_column]
            target_fail = training_data[[target_column]]

            return features[self.top_10_features], target_fail
        else:
            return features[self.top_10_features]



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
        
        x_train, _ , y_train, _ = train_test_split(features, target.squeeze(), test_size = 0.33, random_state = 42)

        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        scale = n_y0/n_y1  

        self._model.set_params(scale_pos_weight=scale)
        self._model.fit(x_train, y_train)
        
        print(self.MODEL_ROOT_PATH)

        self._model.save_model(self.MODEL_ROOT_PATH)


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
        self._model.load_model(self.MODEL_ROOT_PATH)

        predictions = self._model.predict(features)
        predictions = [1 if y_pred > 0.5 else 0 for y_pred in predictions]

        return predictions