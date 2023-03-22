from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import streamlit as st



model    = joblib.load('model.pkl.z')
pipeline = joblib.load('pipeline_full.pkl.z')



class FeaturesRenamer(BaseEstimator, TransformerMixin):
    """
    Esta classe deve ser colocada diretamente dentro de um Pipeline.
    Quando 'get_feature_names_out()' é chamado ele concatena o nome do transformador ao nome de cada coluna
    que passará pela transformação. Esta classe desfaz isso retornando os nomes originais.
    """
    def __init__(self, original_names):
        if isinstance(original_names, list):
            self.original_names = original_names
        else:
            self.original_names = list(original_names)
    def fit(self, X, y=None):
        return self
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    def transform(self, X, y=None):
        new_names = list()
        for name in list(X.columns):
            for original in self.original_names:
                if name.endswith('__' + original):
                    new_names.append(original)
        return X.set_axis(new_names, axis=1)



class OutliersZScoreReplacer(BaseEstimator, TransformerMixin):
    """
    Substitui os outliers encontrados pelas medianas de cada atributo.
    """
    def fit(self, X, y=None):
        self.mean_std_median = list()
        for name in X.columns:
            mean   = X[name].mean()
            std    = X[name].std()
            median = X[name].median()
            self.mean_std_median.append((mean, std, median))
        return self
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    def transform(self, X, y=None):
        std_unit = 3
        for index, name in enumerate(X.columns):
            mean    = self.mean_std_median[index][0]
            std     = self.mean_std_median[index][1]
            median  = self.mean_std_median[index][2]
            scores  = ((X[name] - mean) / std)
            filter_mask = ((scores < -std_unit) | (scores > std_unit))
            X.loc[filter_mask, name] = median
        return X
    def get_feature_names_out(self):
        pass



@st.cache_data
def prediction(model, year, price, transmission, mileage, fueltype, tax, mpg, enginesize, brand):
  pass



def main():
  html_temp = """
  <div style="background-color:yellow; padding:13px>
  <h1 style="color:black;text-align:center;">Streamlit - Car pricing - ML App</h1>
  </div>
  """
  st.markdown(html_temp, unsafe_allow_html=True)



if __name__ == '__main__':
  main()