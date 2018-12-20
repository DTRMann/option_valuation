from math import log, sqrt, pi, exp
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot

### Parameters
# Starting price of F_T
P = 100
# Strike price
K = 100
# Risk free interest rate
r = .05
# Volatility
sigma = .2
# Drift rate
mu = .05
# Time to maturity
T = 1
# Timestep
dt = .01
# Current time period
t = 0

### Functions
def brownian_motion(P):
    P_1 = P*np.exp((mu - (sigma**2/2))*dt + sigma*np.random.normal(0,np.sqrt(dt)))
    return P_1
	
def d1_d2(P,t):
    d1 = (np.log(P/K) + (r + sigma**2/2)*(T-t)) * (1/(sigma * np.sqrt(T-t)))
    d2 = d1 - sigma*np.sqrt(T-t)
    return d1,d2
	
def call_value(P,t):
    d1,d2 = d1_d2(P,t)
    value = norm.cdf(d1)*P - norm.cdf(d2)*K*np.exp(-r*(T-t)) 
    return value
	
def delta(P,t):
    d1 = d1_d2(P,t)[0]
    return norm.cdf(d1)
	
def simulate(P,t):
    columns = ['Time','Price','Delta','Value']
    sim_data = pd.DataFrame(columns=columns)
    while t <= 1:
        P = brownian_motion(P)
        apnd = pd.DataFrame([[t,P,delta(P,t),call_value(P,t)]],columns=columns)
        sim_data = sim_data.append(apnd)
        t+=dt
        t = round(t,2)
    return sim_data
	
# Use simulate function to generate two scenarios; one where the price rises above the strike price and one where
# it falls below it. Generate variables df_up and df_down
df = simulate(100,0
df_up = df
df_up['Delta_diff'] = df_up['Delta'].diff()
df_down = df
df_down['Delta_diff'] = df_down['Delta'].diff()
up_and_down = pd.merge(df_down,df_up,how='inner',on='Time')

# Rename
up_and_down = up_and_down.rename(columns = {'Price_x':'Price_dn','Delta_x':'Delta_dn','Value_x':'Value_dn','Delta_diff_x':'Delta_diff_dn', 
                              'Price_y':'Price_up','Delta_y':'Delta_up','Value_y':'Value_up','Delta_diff_y':'Delta_diff_up'})

plt.plot(up_and_down['Time'],up_and_down['Delta_up'])
plt.plot(up_and_down['Time'],up_and_down['Delta_dn'])
plt.gca().legend(('P>K at T','P<K at T'))
plt.title('Figure 3')
plt.xlabel('Time to maturity')
plt.ylabel('Option delta')
plt.savefig('Figure_3.jpg')
plt.show()

plt.plot(up_and_down['Time'],up_and_down['Value_up'])
plt.plot(up_and_down['Time'],up_and_down['Value_dn'])
plt.gca().legend(('P>K at T','P<K at T'))
plt.title('Figure 4')
plt.xlabel('Time to maturity')
plt.ylabel('Option value ($)')
plt.savefig('Figure_4.jpg')
plt.show()

bins = np.linspace(-.3, .3,75)

pyplot.hist(up_and_down['Delta_diff_dn'], bins, alpha=0.5, label='P<K at T')
pyplot.hist(up_and_down['Delta_diff_up'], bins, alpha=0.5, label='P>K at T')
pyplot.legend(loc='upper right')
plt.title('Figure 5')
plt.xlabel('Change in delta')
plt.ylabel('Frequency')
plt.savefig('Figure_5.jpg')
pyplot.show()	
