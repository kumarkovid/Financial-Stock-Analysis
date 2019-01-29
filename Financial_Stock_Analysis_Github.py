import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from pandas import Series
from pandas import DataFrame
from io import StringIO
from scipy import stats
from datetime import datetime
from pandas_datareader import DataReader
#PLOTTING 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns
sns.set_style('whitegrid')


tech_list=['AAPL','GOOG','MSFT','AMZN']
end=datetime.now()
start=datetime(end.year-1,end.month,end.day)

for stock in tech_list:
    globals()[stock]=DataReader(stock,'yahoo',start,end)#LHS assigns data to each stock as a dataframe (REMEMBER)

print(AAPL.describe())
AAPL.info()
AAPL['A1dj Close'].plot(legend=True,figsize=(10,4))
AAPL['Volume'].plot(legend=True,figsize=(10,4))

ma_day=[10,20,50]

for ma in ma_day:
    column_name='MA for %s days' %(str(ma))
    AAPL[column_name]=AAPL['Adj Close'].rolling(ma).mean()

AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(10,4))


AAPL['Daily Return']=AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(legend=True,figsize=(10,4),marker='o',linestyle='--')#Daily return
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')
#OR
AAPL['Daily Return'].hist(bins=100,color='purple')
closing_df=DataReader(tech_list,'yahoo',start,end)['Adj Close']
print(closing_df.head())
tech_rets=closing_df.pct_change()
print(tech_rets.head())

sns.jointplot('GOOG','GOOG',tech_rets,kind='scatter',color='seagreen')
#COMPARE TO ITSELF
sns.jointplot('GOOG','MSFT',tech_rets,kind='scatter',color='seagreen')
sns.pairplot(tech_rets.dropna())
contol the figure
returns_fig=sns.PairGrid(tech_rets.dropna(),)
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)

returns_fig=sns.PairGrid(closing_df.dropna(),)
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)
#WE CAN SEE GOOD CORRELATION BETWEEN BETWEEN (MSFT&AMZN)

#PLOT THE CORRELATION TO SEE IF WE GUESSED CORRECTLY
corr1=tech_rets.dropna().corr()
corr2=closing_df.dropna().corr()
print(corr2)
corr1.dropna().plot(kind='bar',subplots=False)#GOOG and MSFT have highest CORRELATION
corr2.dropna().plot(kind='bar',subplots=False)#MSFT and AMZN have highest CLOSING CORRELATION
mask=np.zeros_like(corr2)
mask[np.triu_indices_from(mask)]=True
sns.heatmap(corr2,cmap=sns.diverging_palette(256,0,sep=80,n=7,as_cmap=True),annot=True,mask=mask)


rets=tech_rets.dropna()
area=np.pi*50
plt.scatter(rets.mean(),rets.std(),s=area)
plt.ylim([0.015,0.030])
plt.xlim([-0.001,0.003])
plt.xlabel('Expexted return')
plt.ylabel('Risk')

for label,x,y in zip(rets.columns,rets.mean(),rets.std()):
    plt.annotate(
            label,
            xy=(x,y), xytext=(50,50),
            textcoords='offset points', ha='right', va='bottom',
            arrowprops=dict(arrowstyle='-',color='red',connectionstyle='arc3,rad=-0.3'))
#CHECK HOW TO MAKE THESE LOOK BETTER???? (WORK ON IT)

sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')
print(tech_rets['AAPL'].quantile(0.05))
print(tech_rets['AMZN'].quantile(0.01))
print(tech_rets['MSFT'].quantile(0.01))
print(tech_rets['GOOG'].quantile(0.01))
WITH 95 PC CONFIDENCE THE LOSS WONT BE MORE THAN rets*100 at 95% of days 

#MONTE CARLO METHOD{run various simulations, then find how risky a stock is}
days=365
dt=1/days
mu=rets.mean()['GOOG']
sigma=rets.std()['GOOG']

def stock_monte_carlo (start_price,days,mu,sigma):
    #define price array
    price=np.zeros(days)
    price[0]=start_price
    #Shock and Drift
    shock=np.zeros(days)
    drift=np.zeros(days)
    # Run price array for number of days
    for x in range(1,days):
        # Calculate Schock, random normal to choose E0 value
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        # Calculate Drift, s(mu)(delta t)
        drift[x] = mu * dt
        # Calculate Price, S(sigma)(E0)(sqrt(delta t))
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price
start_price=GOOG['Open'][0]

for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))

plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for Google')


runs=100000
simulations=np.zeros(runs)

for run in range(runs):
    simulations[run]=stock_monte_carlo(start_price,days,mu,sigma)[days-1]
#we are gathering the end points of dataset(closing price) for 10K runs

q=np.percentile(simulations,1)#numpy to fit 99 pc of the values in our output...????
plt.hist(simulations,bins=200)
plt.figtext(0.6,0.8,s='Start price:$%.2f' %start_price)
plt.figtext(0.6,0.7,s='Mean Final Price:$%.2f' %simulations.mean())
plt.figtext(0.6,0.6,s='VAR(0.99): $%.2f' %(start_price-q,))#Value at Risk
plt.figtext(0.15,0.6,s='1pc quantile: $%.2f' %q)#quantile plot
plt.axvline(x=q,linewidth=4,color='r')
plt.title(u'Final Price distribution of google price after %s days' %days,weight='bold')










