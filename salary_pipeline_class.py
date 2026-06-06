import numpy as np
import pandas as pd
from sklearn.base     import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class SalaryPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transformer que convierte un DataFrame con columnas originales
    en la matriz de features lista para el modelo.
    """

    EXP_MAP  = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    SIZE_MAP = {'S': 0,  'M': 1,  'L': 2}

    def __init__(self, te_maps: dict, global_mean: float):
        self.te_maps     = te_maps
        self.global_mean = global_mean

    def fit(self, X, y=None):
        return self

    def _te(self, series: pd.Series, col: str) -> pd.Series:
        return series.map(lambda v: self.te_maps[col].get(v, self.global_mean))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = X.copy()
        out = pd.DataFrame()
        out['work_year']        = pd.to_numeric(df['work_year'], errors='coerce').fillna(2023)
        out['exp_encoded']      = df['experience_level'].map(self.EXP_MAP).fillna(1)
        out['size_encoded']     = df['company_size'].map(self.SIZE_MAP).fillna(1)
        out['remote_ratio']     = pd.to_numeric(df['remote_ratio'], errors='coerce').fillna(0)
        out['is_fulltime']      = (df['employment_type'] == 'FT').astype(int)
        out['is_remote']        = (df['remote_ratio'].astype(str) == '100').astype(int)
        out['is_us_company']    = (df['company_location'] == 'US').astype(int)
        out['is_us_employee']   = (df['employee_residence'] == 'US').astype(int)
        out['same_country']     = (df['company_location'] == df['employee_residence']).astype(int)
        out['job_title_te']     = self._te(df['job_title'],          'job_title')
        out['comp_location_te'] = self._te(df['company_location'],   'company_location')
        out['emp_residence_te'] = self._te(df['employee_residence'], 'employee_residence')
        return out.values.astype(float)


class SalaryPipeline(Pipeline):
    """
    Subclase de sklearn Pipeline que aplica expm1 al resultado
    para devolver el salario en USD (no en escala logarítmica).
    """

    def predict(self, X, **kwargs):
        log_pred = super().predict(X, **kwargs)
        return np.expm1(log_pred)
