import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class KolmogorovGaborPolynomial:
    """
    Класс для построения модели полинома Колмогорова-Габора.

    Атрибуты:
    ----------
    models_dict : dict
        Словарь для хранения обученных моделей для каждой итерации.

    partial_polynomial_df : DataFrame
        DataFrame для хранения промежуточных результатов в процессе обучения.

    stop : int
        Количество итераций для обучения модели.
    """

    def __init__(self):
        """
        Инициализирует класс KolmogorovGaborPolynomial.
        """
        self.models_dict = {}

    def fit(self, X, Y, stop=None):
        """
        Обучение модели на основе входных данных.

        Параметры:
        ----------
        X : DataFrame
            Входные данные (признаки).
        Y : DataFrame или Series
            Целевые значения.
        stop : int, optional
            Количество итераций для обучения модели (по умолчанию None, используется число признаков).

        Возвращает:
        ----------
        model : LinearRegression
            Обученная модель из последней итерации.
        """
        if stop is None:
            stop = len(X.columns)
        self.stop = stop

        # Инициализация начальной модели (первая итерация)
        model = LinearRegression()
        model.fit(X, Y)
        predictions = model.predict(X)

        # Сохранение начальных предсказаний из первой модели
        self.partial_polynomial_df = pd.DataFrame(index=Y.index)
        self.partial_polynomial_df['Y'] = Y.values.flatten()
        self.partial_polynomial_df['Y_pred'] = predictions.flatten()

        # Сохранение модели из первой итерации
        self.models_dict['1'] = model

        for i in range(2, stop + 1):
            # Добавление степеней начальных предсказаний из первой модели
            self.partial_polynomial_df[f'Y_pred^{i}'] = (predictions ** i).flatten()
            self.partial_polynomial_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.partial_polynomial_df.fillna(0, inplace=True)

            # Обучение новой модели с дополнительными признаками
            model = LinearRegression()
            X_new = self.partial_polynomial_df.drop(columns='Y')
            model.fit(X_new, Y)

            self.models_dict[str(i)] = model

        return self.models_dict[str(stop)]

    def predict(self, X, stop=None):
        """
        Предсказание на основе обученной модели.

        Параметры:
        ----------
        X : DataFrame
            Входные данные (признаки).
        stop : int, optional
            Количество итераций для предсказания (по умолчанию None, используется self.stop).

        Возвращает:
        ----------
        predictions : ndarray
            Предсказанные значения.
        """
        if stop is None:
            stop = self.stop

        # Начальные предсказания на основе первой модели
        model = self.models_dict['1']
        predictions = model.predict(X)

        if stop == 1:
            return predictions

        # Сохранение начальных предсказаний из первой модели и добавление степеней для каждой итерации
        predict_polynomial_df = pd.DataFrame(index=X.index)
        predict_polynomial_df['Y_pred'] = predictions.flatten()

        for i in range(2, stop + 1):
            # Добавление степеней начальных предсказаний из первой модели
            predict_polynomial_df[f'Y_pred^{i}'] = (predictions ** i).flatten()
            predict_polynomial_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            predict_polynomial_df.fillna(0, inplace=True)

            # Использование соответствующей модели для текущей итерации
            model = self.models_dict[str(i)]

        # Возврат итоговых предсказаний из последней итерации
        final_predictions = model.predict(predict_polynomial_df)

        # Возвращаем DataFrame с индексами
        # return final_predictions
        return pd.DataFrame({'Predictions': final_predictions}, index=range(len(final_predictions)))


