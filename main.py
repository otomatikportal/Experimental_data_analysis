import pandas as pd
import numpy as np
from IPython.display import display
import sklearn
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import signal
from scipy.stats import pearsonr
from preprocess_funcs import prpcss
import matplotlib.pyplot as plt



gdpdf = pd.read_csv('Raw_Data\Economical\GDP (CurrentUSD)\gpd_data_incurrentUSD.csv')
gdpdf = gdpdf.drop(['Country Name','Indicator Name','Indicator Code'], axis=1)

co2ktdf = pd.read_csv('Raw_Data\Environmental\Co2_emissionsKT\CO2_Emissions_ByCountry_KT.csv')
co2ktdf = co2ktdf.drop(['Series Name','Series Code','Country Name'], axis=1)
col_names = co2ktdf.columns.tolist()
col_names[1:] = [name[:4] for name in col_names[1:]]
co2ktdf.columns = col_names

lawindex = pd.read_csv('Raw_Data\Legal\Rule of Law Index\Law_Index.csv')
lawindex = lawindex.drop(['Country Name','Indicator Name','Indicator Code'], axis=1)

""""
#Function to select and interpolate data
def preprocess1(df, country, yeartostart):
    df = df.loc[df['Country Code'] == country, yeartostart:]
    df = df.loc[1:].interpolate(method='linear', axis=1)
    return df

lawinterpolated = preprocess1(lawindex, 'USA', '2000')
print(lawinterpolated)

#Function to scale data (and turns it into array)
def preprocess2(df_interpolated):
    scaler = MinMaxScaler()
    df_interpolated = scaler.fit_transform(df_interpolated.values.reshape(-1, 1))
    return df_interpolated

lawscaled = preprocess2(lawinterpolated)
print(lawscaled)
gdpinterpolated = preprocess1(gdpdf, 'USA', '2000')
gdpscaled = preprocess2(gdpinterpolated)
print(gdpscaled)

gdpdetrended = signal.detrend(gdpscaled, type='linear')
print(gdpdetrended)

gdpdetrended = (gdpdetrended - gdpdetrended.min()) / (gdpdetrended.max() - gdpdetrended.min())

gdpdetrended = gdpdetrended.flatten()
lawscaled = lawscaled.flatten()

corr, _ = pearsonr(gdpdetrended, lawscaled)
print('Pearsons correlation: %.3f' % corr)

import matplotlib.pyplot as plt

plt.plot(gdpdetrended, label='Detrended GDP')
plt.plot(lawscaled, label='Law Index')
plt.xlabel('Time')
plt.ylabel('Detrended GDP')
plt.title(f'normalised gdp and law index (pearson r value{corr:.3f})')
plt.legend()
plt.show()

# combined2darray = np.vstack((lawscaled, gdpscaled)).T
# gc_res = grangercausalitytests(combined2darray, 2)
# print(gc_res)

# Can't conduct granger causality test due to insufficient observations!
"""
gdpusa   = prpcss(gdpdf, 'USA', '2000')
lawusa   = prpcss(lawindex, 'USA', '2000')
co2ktusa = prpcss(co2ktdf, 'USA', '2000')

plt.plot(gdpusa, label='GDP')
plt.plot(lawusa, label='Law Index')
plt.plot(co2ktusa, label='CO2 Emissions')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('GDP, Law Index, and CO2 Emissions')
plt.legend()
plt.show()

