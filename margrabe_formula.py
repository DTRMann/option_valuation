from math import log, sqrt, pi, exp
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

# Prices of P and G at time t<T
Pt = 8
Gt = 10
# Risk-neutral rate of return
r = .05
# Time to maturity
T = 1
rho = 1
## Parameters
hr = .9
sig_1 = 4
sig_2 = (hr*sig_1)

def sig(sig_1,sig_2,rho):
    return sqrt(sig_1**2 + sig_2**2 - 2*sig_1*sig_2*rho)
	
def d_1_2(Pt,Gt,rho,T,sig_1,sig_2):
    d1 = (log(Pt/(hr*Gt)) + (sig(sig_1,sig_2,rho)**2/2)*T)/(sig(sig_1,sig_2,rho)*sqrt(T))
    d2 = d1 - sig(sig_1,sig_2,rho)*sqrt(T)
    return d1,d2
	
def margrabe(Pt,Gt,d1,d2,T):
    return exp(-r*T)*Pt*norm.cdf(d1) - exp(-r*T)*hr*Gt*norm.cdf(d2)
	
def margrabe_run(Pt,Gt,rho,T,sig_1,sig_2):
    d1_d2 = d_1_2(Pt,Gt,rho,T,sig_1,sig_2)
    return margrabe(Pt,Gt,d1_d2[0],d1_d2[1],T)

# Demonstrating option value at T-t=0	
columns = ['T-t','value']
option_value = pd.DataFrame(columns=columns)
for i in range(10):
    time = 1-(i/10)
    time = round(time,2)
    apnd = pd.DataFrame([[time,margrabe_run(Pt,Gt,rho,time,sig_1,sig_2)]],columns=columns)
    option_value = option_value.append(apnd)
	
graph = option_value.plot(x='T-t',y='value')
graph.invert_xaxis()
plt.title('Figure 1')
plt.xlabel('Time to maturity')
plt.ylabel('Extrinsic option value ($)')
plt.savefig('Figure_1.jpg')
plt.show()

# Case where theta=1
Pt = 8
Gt = 10
# Risk-neutral rate of return
r = .05
# Time to maturity
T = .25
rho = 1
## Parameters
hr = 1
sig_1 = 4
sig_2 = (hr*sig_1)

columns = ['sigma_2','value']
option_value = pd.DataFrame(columns=columns)
hr1 = 1.5
sig_2 = hr*sig_1

for i in range(10):
    hr = hr1-.05*i
    sig_2 = hr*sig_1
    apnd = pd.DataFrame([[sig_2,margrabe_run(Pt,Gt,rho,T,sig_1,sig_2)]],columns=columns)
    option_value = option_value.append(apnd)

graph = option_value.plot(x='sigma_2',y='value')
graph.invert_xaxis()
plt.title('Figure 2')
plt.xlabel('σ Gt*hr')
plt.ylabel('Extrinsic option value ($)')
plt.savefig('Figure_2.jpg')
plt.show()

	






