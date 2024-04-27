import pandas as pd
import numpy as np
from IPython.display import display
import sklearn
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import grangercausalitytests


gdpdf = pd.read_csv('Raw_Data\Economical\GDP (CurrentUSD)\gpd_data_incurrentUSD.csv')
gdpdf = gdpdf.drop(['Country Name','Indicator Name','Indicator Code'], axis=1)

co2ktdf = pd.read_csv('Raw_Data\Environmental\Co2_emissionsKT\CO2_Emissions_ByCountry_KT.csv')
co2ktdf = co2ktdf.drop(['Series Name','Series Code','Country Name'], axis=1)
col_names = co2ktdf.columns.tolist()
col_names[1:] = [name[:4] for name in col_names[1:]]
co2ktdf.columns = col_names

lawindex = pd.read_csv('Raw_Data\Legal\Rule of Law Index\Law_Index.csv')
lawindex = lawindex.drop(['Country Name','Indicator Name','Indicator Code'], axis=1)
print(lawindex)

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

from scipy import signal

detrended = signal.detrend(gdpscaled, type='linear')
print(detrended)

detrended = (detrended - detrended.min()) / (detrended.max() - detrended.min())

import matplotlib.pyplot as plt
plt.plot(detrended)
plt.plot(gdpscaled)
plt.xlabel('Time')
plt.ylabel('Detrended GDP')
plt.title('Detrended GDP over Time')
plt.show()



# combined2darray = np.vstack((lawscaled, gdpscaled)).T
# gc_res = grangercausalitytests(combined2darray, 2)
# print(gc_res)

# Can't conduct granger causality test due to insufficient observations!