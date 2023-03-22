from sklearn.base     import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle as pkl
import streamlit as st



class FeaturesRenamer(BaseEstimator, TransformerMixin):
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



model    = pkl.load(open('model_rforest.pkl', 'rb'))
pipeline = pkl.load(open('pipeline_transformer.pkl', 'rb'))


@st.cache_data
def prediction(model, year, price, transmission, mileage, fueltype, tax, mpg, enginesize, brand):
    input_data = {'model': model, 
                'year': year, 
                'price': price, 
                'transmission': transmission, 
                'mileage': mileage, 
                'fueltype': fueltype, 
                'tax': tax, 
                'mpg': mpg, 
                'enginesize': enginesize, 
                'brand': brand}
    X = pipeline.transform(pd.DataFrame(data=input_data, index=range(1)))
    print(X)
    print(str(type(model)))
    #return model.predict(X)
    return str(model)



def main():
  html_temp = """<div style="background-color:yellow; padding:13px>
                     <h1 style="color:black;text-align:center;">Streamlit - Car pricing - ML App</h1>
                 </div>"""
  st.markdown(html_temp, unsafe_allow_html=True)

  brand = st.selectbox('Marca', ('skoda', 'ford', 'vw', 'bmw', 'vauxhall', 'audi', 'toyota', 'mercedes', 'hyundi'))
  model = st.selectbox('Modelo', (' Scala', ' C-MAX', ' Mondeo', ' Golf', ' 1 Series', ' 3 Series',' Mokka X', ' A4', ' Tiguan', ' A3', ' A6', ' Aygo', ' Q3', ' Beetle', ' A1', ' Focus', ' Fiesta', ' GLA Class', ' Astra', ' C-HR', ' 6 Series', ' EcoSport', ' A5', ' Corsa', ' Viva', ' TT', ' GLE Class', ' 4 Series', ' IX20', ' RAV4', ' GLC Class', ' Kuga', ' CL Class', ' 5 Series', ' Grand C-MAX', ' Kona', ' I30', ' I20', ' GT86', ' C Class', ' Octavia', ' Polo', ' Jetta', ' A Class', ' Verso', ' Crossland X', ' Zafira', ' Insignia', ' Antara', ' Grand Tourneo Connect', ' B Class', ' Up', ' Corolla', ' E Class', ' X6', ' I10', ' X2', ' Passat', ' X1', ' Tucson', ' 2 Series', ' Golf SV', ' Yeti Outdoor', ' T-Roc', ' Mokka', ' Ka+', ' Superb', ' Grandland X', ' Kodiaq', ' Q7', ' Citigo', ' V Class', ' Q5', ' Rapid', ' Santa Fe', ' Q2', ' CC', ' Puma', ' Edge', ' Prius', ' Arteon', ' Yaris', ' Auris', ' T-Cross', ' California', ' Touran', ' Touareg', ' Fabia', ' Combo Life', ' Adam', ' SLK', ' SL CLASS', ' Sharan', ' X4', ' S-MAX', ' GL Class', ' S Class', ' I800', ' Karoq', ' X3', ' CLS Class', ' Caddy Maxi Life', ' Scirocco', ' SQ5', ' Hilux', ' Vivaro', ' Ioniq', ' Vectra', ' X-CLASS', ' Kamiq', ' M2', ' KA', ' 7 Series', ' IQ', ' A7', ' X5', ' Galaxy', ' A8', ' RS6', ' Tourneo Custom', ' B-MAX', ' Tiguan Allspace', ' Tourneo Connect', ' GTC', ' Caravelle', ' Caddy Life', ' GLB Class', ' Getz', ' X7', ' Meriva', ' M5', ' M Class', ' Z4', ' Land Cruiser', ' Avensis', ' Agila', ' M4', ' Q8', ' RS3', ' I40', ' IX35', ' Yeti', ' RS5', ' Amarok', ' Ampera', ' i3', ' Shuttle', ' M3', ' Zafira Tourer', ' PROACE VERSO', ' 8 Series', ' G Class', ' CLA Class', ' S4', ' GLS Class', ' RS7', ' Mustang', ' SQ7', ' Cascada', ' Supra', ' R8', ' S3', ' RS4', ' Camry', ' Fusion', ' S8', ' Caddy Maxi', ' Roomster', ' Z3', ' R Class', ' Eos', ' Kadjar', ' i8', ' Fox', ' Caddy', ' Transit Tourneo', '200', ' S5', ' Tigra', ' Urban Cruiser', ' Ranger', ' CLK', ' Amica', ' Escort', ' Terracan', ' M6', ' Veloster', ' Accent', '180', ' CLC Class', ' Verso-S', ' A2'))
  year = st.selectbox('Ano', (2019, 2016, 2015, 2017, 2018, 2012, 2020, 2013, 2014, 2008, 2005, 2009, 2003, 2007, 2011, 2010, 2006, 2004, 1999, 2000, 2001, 2002, 1998, 2060, 1997, 1995, 1996, 1970))
  transmission = st.selectbox('Transmissao', ('Automatic', 'Manual', 'Semi-Auto', 'Other'))
  enginesize = st.selectbox('Motor', (1. , 2. , 1.4, 1.6, 3. , 2.1, 1.8, 1.5, 2.5, 1.2, 1.1, 4. , 1.7, 2.2, 1.3, 1.9, 2.9, 2.4, 2.3, 0. , 6.2, 3.2, 3.5, 4.7, 4.4, 2.7, 6. , 2.8, 5. , 5.2, 5.5, 6.6, 4.2, 0.6, 2.6, 3.6, 3.7, 4.3, 5.4, 6.3))
  fueltype = st.selectbox('Combustível', ('Petrol', 'Diesel', 'Hybrid', 'Other', 'Electric'))
  mileage = st.number_input('Kilometragem')
  tax = st.number_input('Tax')
  mpg = st.number_input('Km/Litro')
  price = 0

  if st.button('Definir preço'):
    result = prediction(model, year, price, transmission, mileage, fueltype, tax, mpg, enginesize, brand)
    st.success(f'O preço sugerido é R${result}')



if __name__ == '__main__':
  main()
