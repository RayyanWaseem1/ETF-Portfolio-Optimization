import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
os.makedirs('plots', exist_ok=True)
plt.style.use('dark_background')

#Tickers
tickers = [
    'SPY', 'QQQ', 'VTI', 'AGG',
    'GLD', 'VNQ', 'XLF', 'XLE',
    'XLK', 'XLI', 'XLRE', 'EEM'
]

etf_map = {'SPY':'Equity (Large Cap)', 'QQQ': 'Equity (Large Cap)', 'VTI': 'Equity (Total Market)', 'AGG': 'Fixed Income',
           'GLD': 'Commodity', 'VNQ': 'Real Estate', 'XLF': 'Equity (Financials)', 'XLE': 'Equity (Energy)',
           'XLK': 'Equity (Technology)', 'XLI': 'Equity (Industrials)', 'XLRE': 'Equity (Real Estate)', 'EEM': 'Equity (Emerging Markets)',
}

#Downloading Historical Data
batches = [tickers[i:i+4] for i in range(0, len(tickers), 4)]
all_data = []

for batch in batches:
  batch_data = yf.download(batch, start='2015-01-01', end='2025-01-01', auto_adjust=True)
  close_prices = batch_data['Close'] if isinstance(batch_data.columns, pd.MultiIndex) else pd.DataFrame(batch_data['Close'])
  all_data.append(close_prices)

price_data = pd.concat(all_data, axis=1)
price_data = price_data.ffill().dropna(axis=1, thresh=len(price_data) * 0.95)

returns = price_data.pct_change().dropna()
avg_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

#Correlation Heatmap
correlation_matrix = returns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidth=0.5,
    square=True,
    cbar_kws={'label': 'Correlation Coefficient'},
    annot_kws={'size': 8}
)

plt.title('Asset Correlation Heatmap (Daily Returns)', fontsize=16, color='white', pad=15)
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')
plt.tight_layout()
plt.savefig('plots/asset_correlation_heatmap.png', dpi=300)
plt.show()

#Defining Constraints
all_classes = set(etf_map.values())
unconstrained_caps = {cls: 1.0 for cls in all_classes}
constraints = {
    'conservative': {
        'asset_max': 0.15,
        'class_max': {
            'Equity (Large Cap)': 0.4,
            'Equity (Total Market)': 0.4,
            'Equity (Financials)': 0.2,
            'Equity (Energy)': 0.2,
            'Equity (Technology)': 0.2,
            'Equity (Industrials)': 0.2,
            'Equity (Real Estate)': 0.2,
            'Equity (Emerging Markets)': 0.2,
            'Fixed Income': 0.2,
            'Commodity': 0.2,
        }
        },
    'aggressive': {
        'asset_max': 0.25,
        'class_max': {
            'Equity (Large Cap)': 0.8,
            'Equity (Total Market)': 0.8,
            'Equity (Financials)': 0.3,
            'Equity (Energy)': 0.3,
            'Equity (Technology)': 0.3,
            'Equity (Industrials)': 0.3,
            'Equity (Real Estate)': 0.3,
            'Equity (Emerging Markets)': 0.3,
            'Fixed Income': 0.2,
            'Commodity': 0.2,
        }
        },
    'unconstrained': {
        'asset_max': 1.0,
        'class_max': unconstrained_caps
    }
    }

#Efficient Frontier/Monte Carlo Simulation
num_portfolios = 1000
all_metrics_by_strategy = {}
all_weights_by_strategy = {}
np.random.seed(42)

mean_arr = avg_returns.values
cov_matrix_arr = cov_matrix.values
asset_names = price_data.columns.tolist()
class_map = etf_map

for strategy, constraint in constraints.items():
  metrics_list = []
  weights_list = []

  for _ in range(num_portfolios):
    #1) Generate Random Weights and Normalize
    weights = np.random.random(len(asset_names))
    weights /= weights.sum()

    #2)If Not Unconstrained, Enforce Caps
    if strategy != 'unconstrained':
      #2a) Asset Level Cap
      if np.any(weights > constraint['asset_max']):
        continue

      #2b) Class Level Cap
      #Aggregate Weights by Class
      class_sums = {}
      for i, ticker in enumerate(asset_names):
        cls = class_map[ticker]
        class_sums[cls] = class_sums.get(cls, 0.0) + weights[i]

      #Check each class against its cap
      violated = False
      for cls, cap in constraint['class_max'].items():
        if class_sums.get(cls, 0.0) > cap:
          violated = True
          break

      if violated:
        continue

    #3)Compute Performance Metrics
    portfolio_return = weights.dot(mean_arr)
    portfolio_volatility = np.sqrt(weights @ cov_matrix_arr @ weights.T)
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

    #4)Store
    metrics_list.append((portfolio_return, portfolio_volatility, sharpe_ratio))
    weights_list.append(weights)


  #convert to DataFrame
  df_metrics = pd.DataFrame(metrics_list, columns=['Return', 'Volatility', 'Sharpe Ratio'])
  df_weights = pd.DataFrame(weights_list, columns=asset_names)

  all_metrics_by_strategy[strategy] = df_metrics
  all_weights_by_strategy[strategy] = df_weights

#Plotting Efficient Frontier Comparison
plt.figure(figsize=(10,6))
colors = {
    'unconstrained': 'blue',
    'conservative': 'green',
    'aggressive': 'red'
}

for strategy, metrics_list in all_metrics_by_strategy.items():
  #Scatter the portfolios for this strategy
  plt.scatter(
      metrics_list['Volatility'],
      metrics_list['Return'],
      c=metrics_list['Sharpe Ratio'],
      cmap='RdYlGn',
      s=15,
      label=f'{strategy.capitalize()} Portfolios',
      alpha=0.4,
  )

  #Highlight Max Sharpe Portfolio
  max_idx = metrics_list['Sharpe Ratio'].idxmax()
  best = metrics_list.loc[max_idx]
  plt.scatter(
      best['Volatility'],
      best['Return'],
      marker = '*',
      s=250,
      edgecolors='white',
      linewidths = 1.5,
      color = colors[strategy],
      label = f'{strategy.capitalize()} Max Sharpe Portfolio'
  )

plt.title('Efficient Frontier: Comparison of Strategies')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')
plt.legend(loc = 'upper left')
plt.grid(True, linestyle = '--', alpha = 0.7)
plt.tight_layout()
plt.savefig('plots/efficient_frontier_comparison.png', dpi=300)
plt.show()

#Pie Charts for Sector Allocation
for strategy in constraints.keys():
  #1)Finding Index of Max Sharpe Portfolio
  df_metrics = all_metrics_by_strategy[strategy]
  best_idx = df_metrics['Sharpe Ratio'].idxmax()

  #2)Pulling Corresponding Weights
  df_weights = all_weights_by_strategy[strategy]
  optimal_weights = df_weights.loc[best_idx]

  #3)Aggregating Weights by Asset Class
  class_alloc = {}
  for ticker, weight in optimal_weights.items():
    cls = etf_map[ticker]
    class_alloc[cls] = class_alloc.get(cls, 0.0) + weight

  #4)Sort in Descending Order by Weight
  clas_alloc = dict(
      sorted(class_alloc.items(), key=lambda kv: kv[1], reverse=True)
  )

  #5)Plotting Pie Chart
  plt.figure(figsize=(6,6))
  colors = plt.cm.tab20(np.linspace(0, 1, len(class_alloc)))
  plt.pie(
      class_alloc.values(),
      labels=[f'{c}\n{v:.1%}' for c, v in class_alloc.items()],
      autopct=None,
      startangle=140,
      colors=colors,
      textprops={'color':'white', 'fontsize':12},
      wedgeprops={'linewidth':1, 'edgecolor':'white'}
  )

  plt.title(
      f'{strategy.capitalize()} Strategy \n Asset-Class Allocation',
      color = 'white', fontsize=16, pad=20
  )

  plt.tight_layout()
  plt.savefig(f'plots/{strategy}_class_allocation.png', dpi=300)
  plt.show()

#Side by Side bar plot of weights
colors = {
    'unconstrained': 'blue',
    'conservative': 'green',
    'aggressive': 'red'
}

#1) Extract Each Strategy's Max Sharpe Weight Vector
optimal_weights = {}
for strat, metrics in all_metrics_by_strategy.items():
  best_idx = metrics['Sharpe Ratio'].idxmax()
  optimal_weights[strat] = all_weights_by_strategy[strat].loc[best_idx]

#2) Building Dataframe: Rows are Tickers, Columns are Strategies
weight_comparison_df = pd.DataFrame(optimal_weights)
weight_comparison_df = weight_comparison_df.sort_index()

#3) Plotting chart
ax = weight_comparison_df.plot(
    kind='bar',
    figsize=(14,8),
    color = [colors[strat] for strat in weight_comparison_df.columns],
    edgecolor='black',
    width=0.8,
)

#4) Formatting
ax.set_title('Optimal ETF Weights by Strategy', fontsize = 16, pad=12),
ax.set_xlabel('ETF Ticker', fontsize=14),
ax.set_ylabel('Portfolio Weight', fontsize=14)
ax.legend(title = 'Strategy', fontsize=12, title_fontsize=13, loc = 'upper right')
ax.grid(axis='y', linestyle= '--', alpha = 0.5)
plt.xticks(rotation = 45, ha = 'right')
plt.tight_layout()

plt.savefig('plots/optimal_weights_comparison.png', dpi=300)
plt.show()

