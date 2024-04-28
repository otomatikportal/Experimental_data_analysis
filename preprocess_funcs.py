import pandas as pd
import numpy as np
from IPython.display import display
import sklearn
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import signal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

gdpdf = pd.read_csv('Raw_Data\Economical\GDP (CurrentUSD)\gpd_data_incurrentUSD.csv')
gdpdf = gdpdf.drop(['Country Name','Indicator Name','Indicator Code'], axis=1)

lawindex = pd.read_csv('Raw_Data\Legal\Rule of Law Index\Law_Index.csv')
lawindex = lawindex.drop(['Country Name','Indicator Name','Indicator Code'], axis=1)


# Note that y axis of df should maximum 1

#DATA SELECTION AND INTERPOLATION
def prpcss(df, country, yeartostart):
    df = df.loc[df['Country Code'] == country, yeartostart:]
    array1 = df.values.flatten()


    # Create an array 'x' representing the indices of 'array1'
    x = np.arange(len(array1))

    # Get the indices of the non-missing values in 'array1'
    valid_indices = np.where(~np.isnan(array1))[0]

    # Get the non-missing values in 'array1'
    valid_values = array1[valid_indices]

    # Perform linear interpolation
    interpolated_array = np.interp(x, valid_indices, valid_values)
    scaledinterpolatedarray = MinMaxScaler().fit_transform(interpolated_array.reshape(-1, 1))

    return interpolated_array, scaledinterpolatedarray
"""
interpolatedarray, scaled = prpcss(lawindex, 'USA', '2000')
plt.plot(interpolatedarray)
plt.title('Interpolated Array')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

plt.plot(scaled)
plt.title('Scaled Interpolated Array')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
"""


 





