
#computation
import pandas as pd
import numpy as np
import scipy as sp

#visualization
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

#data usage
import databento as db

chunk_size = 10000
#importing the data in chunks
df_iter = db.DBNStore.from_file(r"C:\Users\matthewzz\Desktop\visualize\options1.dbn").to_df(count = chunk_size)

df = next(df_iter)
print(df.shape)

df

df = df[["open","high","low","close","volume","symbol", "instrument_id"]]
df.reset_index(inplace = True)

df

df['put-call'] = df['symbol'].str[12]                                                 #Creating data flag differentiaing Calls vs Puts
df["expiration_datetime"] = pd.to_datetime(df['symbol'].str[6:12], format='%y%m%d')   #Convert format to DateTime
df["expiration_datetime"] = df["expiration_datetime"] + pd.Timedelta(hours=17)        #Adjusting the datetime to be 5 PM EST
df["ts_event"] = pd.to_datetime(df["ts_event"]).dt.tz_localize(None)
df["r"] = 0.045                                                                       #approx of risk-free-rate
df["S_0"] = 104                                                                       #approximation of current spot price
df["tau"] = (df["expiration_datetime"] - df["ts_event"]).dt.total_seconds() / 86400   #calc days to exp


# Extract dollar and decimal parts separately
dollar_part = df['symbol'].str[13:18]
decimal_part = df['symbol'].str[18:]
# Combine the parts and convert to float
df["K"] = (dollar_part + '.' + decimal_part).astype(float)
df["symbol_raw"] =df['symbol'].str[0:4]

df = df[["ts_event", "close",  "put-call","expiration_datetime", "r", "S_0", "tau", "K","instrument_id"]]

df

from statistics import NormalDist
import math

def BS_call_calc(S_0, K, sig, tau, r):
  d_plus = (math.log(S_0 / K) + (r + (1/2)*(math.pow(sig,2))*(tau))) / (sig*math.sqrt(tau))
  d_minus = (math.log(S_0 / K) + (r - (1/2)*(math.pow(sig,2))*(tau))) / (sig*math.sqrt(tau))

  return S_0 * NormalDist().cdf(d_plus) - K * math.exp(-r*tau) * NormalDist().cdf(d_minus)

def BS_put_calc(S_0, K, sig, tau, r):
  d_plus = (math.log(S_0 / K) + (r + (1/2)*(math.pow(sig,2))*(tau))) / (sig*math.sqrt(tau))
  d_minus = (math.log(S_0 / K) + (r - (1/2)*(math.pow(sig,2))*(tau))) / (sig*math.sqrt(tau))

  return K * math.exp(-r*tau) * NormalDist().cdf(-1*d_minus) - S_0 * NormalDist().cdf(-1*d_plus)

def BS_vega_calc(S_0, K, sig, tau, r):
  d_plus = (math.log(S_0 / K) + (r + (1/2)*(math.pow(sig,2))*(tau))) / (sig*math.sqrt(tau))
  return S_0 * math.sqrt(tau) * NormalDist().cdf(d_plus)

def BS_IV_calc(C_market, S_0, K, tau, r):
  MAX_ITER = 7
  epsilon = 0.00001
  iter = 0
  tau = tau/365 #converting tau to years for calculations

  m = S_0 / (K*math.exp(-r*tau))
  sig_n = math.sqrt((2*abs(math.log(m))) / tau+0.0001)  #sig_0
  C_BSM = BS_call_calc(S_0, K, sig_n, tau, r)

  while(abs(C_market-C_BSM) > epsilon and iter < MAX_ITER):
    C_BSM = BS_call_calc(S_0, K, sig_n, tau, r)
    vega_BSM = BS_vega_calc(S_0, K, sig_n, tau, r)
    sig_n = sig_n + ((C_market - C_BSM)  /  (vega_BSM+0.00001))

    iter+=1

  return sig_n

def BS_IV_calc_p(P_market, S_0, K, tau, r):
  MAX_ITER = 7
  epsilon = 0.00001
  iter = 0
  tau = tau/365 #converting tau to years for calculations

  m = S_0 / (K*math.exp(-r*tau))

  sig_n = math.sqrt((2*abs(math.log(m))) / tau+0.0001)  #sig_0
  P_BSM = BS_put_calc(S_0, K, sig_n, tau, r)

  while(abs(P_market-P_BSM) > epsilon and iter < MAX_ITER):
    P_BSM = BS_put_calc(S_0, K, sig_n, tau, r)
    vega_BSM = BS_vega_calc(S_0, K, sig_n, tau, r)
    sig_n = sig_n + ((P_market - P_BSM)  /  (vega_BSM+0.00001))

    iter+=1

  return sig_n

df["Moneyness"] = df["S_0"] / df["K"]
df['IV'] = df.apply(lambda row: BS_IV_calc(row['close'], row['S_0'], row['K'], row['tau'], row['r']) if row["put-call"] == "C" else BS_IV_calc_p(row['close'], row['S_0'], row['K'], row['tau'], row['r']), axis=1)
df.reset_index(inplace=True, drop = True)

from datetime import datetime,timedelta
from copy import deepcopy

price_book = {}
price_book[datetime.strptime('2023-04-03 18:00:00', '%Y-%m-%d %H:%M:%S')] = {}

start = df["ts_event"].iloc[0]
curr_timestamp = start

while curr_timestamp < datetime.strptime('2023-04-03 18:56:00', '%Y-%m-%d %H:%M:%S'):
  mini_df = df[df['ts_event'] == curr_timestamp]
  price_book[curr_timestamp] = {}

  if curr_timestamp > start:
    price_book[curr_timestamp] = deepcopy(price_book[curr_timestamp-timedelta(minutes=1)])
    for key in price_book[curr_timestamp]:
      price_book[curr_timestamp][key][0]-=(1/1440)

  keys_to_remove = [key for key in price_book[curr_timestamp] if price_book[curr_timestamp][key][0] < 0]
  for key in keys_to_remove:
      del price_book[curr_timestamp][key]

  for index, row in mini_df.iterrows():
    price_book[curr_timestamp][row["instrument_id"]] = [row["tau"], row["Moneyness"], row["IV"]]

  curr_timestamp += timedelta(minutes=1)

timestamp_list = list(price_book.keys())















fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('IV Curve')
ax.set_xlabel('Time to maturity')
ax.set_ylabel('Moneyness')
ax.set_zlabel('Implied Volatility')
ax.set_xlim(0, 80)
ax.set_ylim(0.8, 1.15)

# Specify the hyperparam for the row you want to plot
hyperparam = 10  # Example index, replace with your actual index

# Extract data for the specific row
if hyperparam < len(timestamp_list):
    TS1 = np.array(list(price_book[df['ts_event'][hyperparam]].values()))
    TS1 = TS1[TS1[:, 2] < 1.3]
    TS1 = TS1[TS1[:, 0] < 100]
    x, y, z = TS1[:, 0], TS1[:, 1], TS1[:, 2]

    # Interpolate data for plotting
    X, Y = np.meshgrid(np.linspace(0, 80, 100), np.linspace(0.8, 1.15, 100))
    Z = griddata((x, y), z, (X, Y), method='cubic')

    # Plot the data
    ax.scatter(x, y, z, c='k', marker='o')  # Plot raw data points
    ax.plot_wireframe(X, Y, Z, color='k')  # Plot interpolated surface

    x_values = np.linspace(0, 50, 7)
    y_values = np.linspace(0.90, 1.15, 7)
    z_value = 0.6  # Constant z value for all points

    # Plotting the scattered red points without forming a diagonal
    for x in x_values:
        for y in y_values:
            ax.scatter(x, y, z_value, c='r', marker='o')

    plt.show()

plt.show()