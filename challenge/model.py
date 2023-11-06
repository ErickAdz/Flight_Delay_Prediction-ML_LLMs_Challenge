from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from typing import Tuple, Union, List
import joblib
from challenge.features import data_agregator
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
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

    def __init__(self):
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Check if we're in training mode (target_column is provided)
        if target_column is not None:
            # Apply data aggregator which also handles the target column
            data = data_agregator(data)

            # Ensure the categorical features are one-hot encoded
            data_encoded = pd.get_dummies(data, columns=['OPERA', 'TIPOVUELO', 'MES'])
            
            # Extract the target
            target = data_encoded[[target_column]]

        else:
            # For prediction, assume data is already preprocessed and just needs one-hot encoding
            # Ensure the categorical features are one-hot encoded
            data_encoded = pd.get_dummies(data, columns=['OPERA', 'TIPOVUELO', 'MES'])

        # Align the features of the input data with the trained model's features
        features = pd.DataFrame(columns=self.top_10_features)
        for feature in self.top_10_features:
            if feature in data_encoded:
                features[feature] = data_encoded[feature]
            else:
                features[feature] = 0

        # Return the processed features and target for training, or just features for prediction
        if target_column is not None:
            return features, target
        else:
            return features


    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, x_test, y_train, y_test = train_test_split(features, target.squeeze(), test_size=0.33, random_state=42)
        
        # Scaling the positive class (assuming binary classification and positive class is 1)
        scale = y_train.value_counts()[0] / y_train.value_counts()[1]
        
        self._model.set_params(scale_pos_weight=scale)
        self._model.fit(x_train, y_train)

        # Save the model
        joblib.dump(self._model, self.MODEL_ROOT_PATH)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            List[int]: predicted targets.
        """
        # Load the model
        loaded_model = joblib.load(self.MODEL_ROOT_PATH)

        # Make predictions
        predictions = loaded_model.predict(features)
        predictions = [1 if y_pred > 0.5 else 0 for y_pred in predictions]

        return predictions