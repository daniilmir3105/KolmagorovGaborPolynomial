import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class KolmogorovGaborPolynomial:
    """
    Class for constructing the Kolmogorov-Gabor polynomial.

    Attributes:
    ----------
    models_dict : dict
        Dictionary for storing trained models.

    partial_polynomial_df : DataFrame
        DataFrame for storing intermediate results during training.

    stop : int
        Number of iterations for training the model.
    """

    def __init__(self):
        """
        Initialize the KolmogorovGaborPolynomial class.
        """
        self.models_dict = {}  # Dictionary for storing models

    def fit(self, X, Y, stop=None):
        """
        Train the model based on input data.

        Parameters:
        ----------
        X : DataFrame
            Input data (features).
        Y : DataFrame or Series
            Target values.
        stop : int, optional
            Number of iterations for training the model (default is None, which means using all features).

        Returns:
        ----------
        model : LinearRegression
            The trained model at the last iteration.
        """
        if stop is None:
            stop = len(X.columns)
        self.stop = stop

        # Initial model (first iteration)
        model = LinearRegression()
        model.fit(X, Y)
        predictions = model.predict(X)

        # Create a DataFrame for storing intermediate results
        self.partial_polynomial_df = pd.DataFrame(index=Y.index)
        self.partial_polynomial_df['Y'] = Y.values.flatten()
        self.partial_polynomial_df['Y_pred'] = predictions.flatten()

        self.models_dict['1'] = model

        for i in range(2, stop + 1):
            # Add new features
            self.partial_polynomial_df[f'Y_pred^{i}'] = (predictions ** i).flatten()

            # Limit prediction values to avoid overflow
            self.partial_polynomial_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.partial_polynomial_df.fillna(0, inplace=True)

            # Train a new model with additional features
            model = LinearRegression()
            X_new = self.partial_polynomial_df.drop(columns='Y')
            model.fit(X_new, Y)
            predictions = model.predict(X_new)

            self.models_dict[str(i)] = model

        return self.models_dict[str(stop)]

    def predict(self, X, stop=None):
        """
        Make predictions based on the trained model.

        Parameters:
        ----------
        X : DataFrame
            Input data (features).
        stop : int, optional
            Number of iterations for prediction (default is None, which means using self.stop value).

        Returns:
        ----------
        predictions : ndarray
            Predicted values.
        """
        if stop is None:
            stop = self.stop

        # Initial predictions
        model = self.models_dict['1']
        predictions = model.predict(X)

        if stop == 1:
            return predictions

        # Create a DataFrame for storing intermediate prediction results
        predict_polynomial_df = pd.DataFrame(index=X.index)
        predict_polynomial_df['Y_pred'] = predictions.flatten()

        for i in range(2, stop + 1):
            # Add new features for prediction
            predict_polynomial_df[f'Y_pred^{i}'] = (predictions ** i).flatten()

            # Limit prediction values to avoid overflow
            predict_polynomial_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            predict_polynomial_df.fillna(0, inplace=True)

            model = self.models_dict[str(i)]
            predictions = model.predict(predict_polynomial_df)

        return predictions
