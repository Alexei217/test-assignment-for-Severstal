"""
DataPreprocessor - класс для базовой очистки и трансформации табличных данных.

Поддерживает:
  - Удаление столбцов с высокой долей пропусков и заполнение оставшихся
  - One-hot encoding категориальных столбцов
  - Min-Max / Z-score нормализацию числовых столбцов
  - Сохранение pipeline для воспроизводимого применения к новым данным
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd


# Вспомогательная структура для хранения состояния pipeline
@dataclass
class PipelineState:
    """Хранит всю информацию, необходимую для повторного применения
    того же набора преобразований к новым данным (transform)."""

    # remove_missing
    dropped_columns: str | None = field(default_factory=list)
    fill_values: dict[str, Any] = field(default_factory=dict)

    # encode_categorical
    onehot_columns: list[str] = field(default_factory=list)
    onehot_new_columns: list[str] = field(default_factory=list)

    # normalize_numeric
    norm_method: str | None = None
    norm_params: dict[str, dict[str, float]] = field(default_factory=dict)
    # для minmax: {col: {"min": …, "max": …}}
    # для std:    {col: {"mean": …, "std": …}}


# Основной класс
class DataPreprocessor:
    """Препроцессор табличных данных на основе pandas DataFrame.

    Пример использования
    --------------------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "x"]})
    >>> prep = DataPreprocessor(df)
    >>> result = prep.fit_transform()
    """

    # Допустимые стратегии заполнения пропусков
    _FILL_STRATEGIES = ("mean", "median", "mode")
    # Допустимые методы нормализации
    _NORM_METHODS = ("minmax", "std")

    def __init__(self, df: pd.DataFrame) -> None:
        self._validate_dataframe(df)
        self._original_df = df.copy()
        self._df = df.copy()
        self._state = PipelineState()
        self._is_fitted = False

    #  Публичные методы
    def remove_missing(
        self,
        threshold: float = 0.5,
        fill_strategy: str = "mean",
    ) -> DataPreprocessor:
        """Удаляет столбцы с долей пропусков > ``threshold`` и заполняет
        оставшиеся пропуски выбранной стратегией.

        Parameters
        ----------
        threshold : float, default 0.5
            Максимально допустимая доля пропущенных значений в столбце
            (от 0.0 до 1.0 включительно).
        fill_strategy : str, default "mean"
            Стратегия заполнения пропусков: ``"mean"`` | ``"median"`` | ``"mode"``.

        Returns
        -------
        self
            Для поддержки цепочечных вызовов (method chaining).
        """
        self._validate_threshold(threshold)
        self._validate_fill_strategy(fill_strategy)

        missing_ratio = self._df.isna().mean()
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

        if cols_to_drop:
            self._df.drop(columns=cols_to_drop, inplace=True)

        self._state.dropped_columns = cols_to_drop

        # Заполняем оставшиеся пропуски
        fill_values: dict[str, Any] = {}
        for col in self._df.columns:
            if self._df[col].isna().sum() == 0:
                continue
            fill_val = self._compute_fill_value(self._df[col], fill_strategy)
            fill_values[col] = fill_val
            self._df[col] = self._df[col].fillna(fill_val)

        self._state.fill_values = fill_values
        return self

    def encode_categorical(self) -> DataPreprocessor:
        """Выполняет one-hot encoding всех категориальных (str / object / category)
        столбцов.

        Returns
        -------
        self
        """
        cat_cols = self._df.select_dtypes(include=["str", "object", "category"]).columns.tolist()

        if not cat_cols:
            warnings.warn("Категориальных столбцов не найдено - encoding пропущен.")
            return self

        self._state.onehot_columns = cat_cols

        self._df = pd.get_dummies(self._df, columns=cat_cols, prefix=cat_cols, dtype=np.uint8)

        # Запоминаем новые столбцы, появившиеся после OHE
        self._state.onehot_new_columns = [
            c for c in self._df.columns if c not in self._original_df.columns
        ]

        return self

    def normalize_numeric(
        self,
        method: Literal["minmax", "std"] = "minmax",
    ) -> DataPreprocessor:
        """Нормализует числовые столбцы.

        Parameters
        ----------
        method : str, default "minmax"
            ``"minmax"`` - Min-Max нормализация в [0, 1].
            ``"std"`` - стандартизация (Z-score): (x - μ) / σ.

        Returns
        -------
        self
        """
        self._validate_norm_method(method)
        self._state.norm_method = method

        num_cols = self._df.select_dtypes(include="number").columns.tolist()
        # Исключаем бинарные OHE-столбцы из нормализации
        num_cols = [c for c in num_cols if c not in self._state.onehot_new_columns]

        if not num_cols:
            warnings.warn("Числовых столбцов для нормализации не найдено.")
            return self

        params: dict[str, dict[str, float]] = {}

        for col in num_cols:
            if method == "minmax":
                col_min = float(self._df[col].min())
                col_max = float(self._df[col].max())
                denom = col_max - col_min
                if denom == 0:
                    warnings.warn(
                        f"Столбец '{col}' имеет нулевой диапазон — "
                        "нормализация даст 0."
                    )
                    self._df[col] = 0.0
                else:
                    self._df[col] = (self._df[col] - col_min) / denom
                params[col] = {"min": col_min, "max": col_max}

            elif method == "std":
                col_mean = float(self._df[col].mean())
                col_std = float(self._df[col].std(ddof=0))
                if col_std == 0:
                    warnings.warn(
                        f"Столбец '{col}' имеет нулевое std — "
                        "стандартизация даст 0."
                    )
                    self._df[col] = 0.0
                else:
                    self._df[col] = (self._df[col] - col_mean) / col_std
                params[col] = {"mean": col_mean, "std": col_std}

        self._state.norm_params = params
        return self

    def fit_transform(
        self,
        threshold: float = 0.5,
        fill_strategy: str = "mean",
        norm_method: Literal["minmax", "std"] = "minmax",
    ) -> pd.DataFrame:
        """Последовательно применяет все три преобразования и возвращает
        обработанный DataFrame.

        Parameters
        ----------
        threshold : float
            Порог для ``remove_missing``.
        fill_strategy : str
            Стратегия заполнения для ``remove_missing``.
        norm_method : str
            Метод нормализации для ``normalize_numeric``.

        Returns
        -------
        pd.DataFrame
        """
        # Сбрасываем внутреннее состояние перед новым fit
        self._df = self._original_df.copy()
        self._state = PipelineState()

        self.remove_missing(threshold=threshold, fill_strategy=fill_strategy)
        self.encode_categorical()
        self.normalize_numeric(method=norm_method)

        self._is_fitted = True
        return self._df.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет ранее сохранённый pipeline к ``новому`` DataFrame.

        Требует предварительного вызова ``fit_transform``.

        Parameters
        ----------
        df : pd.DataFrame
            Новые данные с тем же набором исходных столбцов.

        Returns
        -------
        pd.DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline не обучен. Сначала вызовите fit_transform()."
            )
        self._validate_dataframe(df)
        out = df.copy()

        # 1. Удаление столбцов
        cols_to_drop = [c for c in self._state.dropped_columns if c in out.columns]
        out.drop(columns=cols_to_drop, inplace=True)

        # 2. Заполнение пропусков теми же значениями
        for col, val in self._state.fill_values.items():
            if col in out.columns:
                out[col] = out[col].fillna(val)

        # 3. One-hot encoding
        if self._state.onehot_columns:
            cat_cols_present = [c for c in self._state.onehot_columns if c in out.columns]
            out = pd.get_dummies(out, columns=cat_cols_present,
                                 prefix=cat_cols_present, dtype=np.uint8)

        # 4. Нормализация
        for col, p in self._state.norm_params.items():
            if col not in out.columns:
                continue
            if self._state.norm_method == "minmax":
                denom = p["max"] - p["min"]
                out[col] = 0.0 if denom == 0 else (out[col] - p["min"]) / denom
            elif self._state.norm_method == "std":
                out[col] = 0.0 if p["std"] == 0 else (out[col] - p["mean"]) / p["std"]

        # Гарантируем порядок столбцов как в fit
        out = out.reindex(columns=self._df.columns, fill_value=0)
        return out

    #  Свойства / утилиты для интроспекции
    @property
    def state(self) -> PipelineState:
        """Возвращает объект с информацией о применённых преобразованиях."""
        return self._state

    @property
    def result(self) -> pd.DataFrame:
        """Текущее состояние DataFrame."""
        return self._df.copy()

    def summary(self) -> str:
        """Человекочитаемая сводка о применённых преобразованиях."""
        lines = ["=== DataPreprocessor summary ==="]
        lines.append(f"Удалённые столбцы ({len(self._state.dropped_columns)}): "
                     f"{self._state.dropped_columns}")
        lines.append(f"Заполненные столбцы: {list(self._state.fill_values.keys())}")
        lines.append(f"OHE-исходные столбцы: {self._state.onehot_columns}")
        lines.append(f"OHE-новые столбцы ({len(self._state.onehot_new_columns)}): "
                     f"{self._state.onehot_new_columns}")
        lines.append(f"Метод нормализации: {self._state.norm_method}")
        lines.append(f"Нормализованные столбцы: {list(self._state.norm_params.keys())}")
        lines.append(f"Итоговая форма DataFrame: {self._df.shape}")
        return "\n".join(lines)

    #  Приватные хелперы
    @staticmethod
    def _validate_dataframe(df: Any) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Ожидается pandas DataFrame, получен {type(df).__name__}."
            )
        if df.empty:
            raise ValueError("Передан пустой DataFrame.")

    @staticmethod
    def _validate_threshold(value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"threshold должен быть числом, получен {type(value).__name__}.")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold должен быть в диапазоне [0, 1], получен {value}.")

    @classmethod
    def _validate_fill_strategy(cls, strategy: str) -> None:
        if strategy not in cls._FILL_STRATEGIES:
            raise ValueError(
                f"fill_strategy должен быть одним из {cls._FILL_STRATEGIES}, "
                f"получен '{strategy}'."
            )

    @classmethod
    def _validate_norm_method(cls, method: str) -> None:
        if method not in cls._NORM_METHODS:
            raise ValueError(
                f"method должен быть одним из {cls._NORM_METHODS}, "
                f"получен '{method}'."
            )

    @staticmethod
    def _compute_fill_value(series: pd.Series, strategy: str) -> Any:
        """Вычисляет значение для заполнения пропусков по выбранной стратегии.

        Для числовых столбцов используется mean/median/mode
        в зависимости от параметра strategy.
        Для нечисловых - mode (мода), независимо от запрошенной стратегии.
        """
        if pd.api.types.is_numeric_dtype(series):
            if strategy == "mean":
                return series.mean()
            elif strategy == "median":
                return series.median()
            else:  # mode
                mode_vals = series.mode()
                return mode_vals.iloc[0] if not mode_vals.empty else 0
        else:
            # Для категориальных/строковых всегда мода
            mode_vals = series.mode()
            return mode_vals.iloc[0] if not mode_vals.empty else "UNKNOWN"