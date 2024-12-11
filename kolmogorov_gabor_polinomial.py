import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class KolmogorovGaborPolynomial:
    """
    A class to construct the Kolmogorov-Gabor polynomial model.

    Attributes:
    ----------
    models_dict : dict
        Dictionary to store the trained models for each iteration.

    partial_polynomial_df : DataFrame
        DataFrame to store intermediate results during training.

    stop : int
        Number of iterations to train the model.
    """

    def __init__(self):
        """
        Initializes the KolmogorovGaborPolynomial class.
        """
        self.models_dict = {}

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
            Number of iterations for model training (default is None, which uses all features).

        Returns:
        ----------
        model : LinearRegression
            Trained model from the final iteration.
        """
        if stop is None:
            stop = len(X.columns)
        self.stop = stop

        # Initial model (first iteration)
        model = LinearRegression()
        model.fit(X, Y)
        predictions = model.predict(X)

        # Store initial predictions from the first model
        self.partial_polynomial_df = pd.DataFrame(index=Y.index)
        self.partial_polynomial_df['Y'] = Y.values.flatten()
        self.partial_polynomial_df['Y_pred'] = predictions.flatten()

        # Save the model from the first iteration
        self.models_dict['1'] = model

        for i in range(2, stop + 1):
            # Add powers of the initial predictions from the first model
            self.partial_polynomial_df[f'Y_pred^{i}'] = (predictions ** i).flatten()
            self.partial_polynomial_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.partial_polynomial_df.fillna(0, inplace=True)

            # Train a new model with the additional features
            model = LinearRegression()
            X_new = self.partial_polynomial_df.drop(columns='Y')
            model.fit(X_new, Y)

            self.models_dict[str(i)] = model

        return self.models_dict[str(stop)]

    def predict(self, X, stop=None):
        """
        Predict based on the trained model.

        Parameters:
        ----------
        X : DataFrame
            Input data (features).
        stop : int, optional
            Number of iterations for prediction (default is None, which uses self.stop).

        Returns:
        ----------
        predictions : ndarray
            Predicted values.
        """
        if stop is None:
            stop = self.stop

        # Initial predictions based on the first model
        model = self.models_dict['1']
        predictions = model.predict(X)

        if stop == 1:
            return predictions

        # Store the initial predictions from the first model and add powers for each iteration
        predict_polynomial_df = pd.DataFrame(index=X.index)
        predict_polynomial_df['Y_pred'] = predictions.flatten()

        for i in range(2, stop + 1):
            # Add powers of the initial predictions from the first model
            predict_polynomial_df[f'Y_pred^{i}'] = (predictions ** i).flatten()
            predict_polynomial_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            predict_polynomial_df.fillna(0, inplace=True)

            # Use the corresponding model for the current iteration
            model = self.models_dict[str(i)]

        # Return final predictions from the last iteration
        final_predictions = model.predict(predict_polynomial_df)

        # Возвращаем DataFrame с индексами
        # return final_predictions
        return pd.DataFrame({'Predictions': final_predictions}, index=range(len(final_predictions)))
