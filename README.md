# ETF-Portfolio-Optimization

## Project Overview
In this project, I set out to apply Modern Portfolio Theory to a diversified basket of Exchange-Traded Funds (ETFs) in order to understand and learn how varying levels of risk appetite and portfolio constraints influence the optimal mix of asset classes. Rather than focusing on individual equities or fixed-income securities, I selected twelve broadly representative ETFs – ranging from large-cap equities (e.g. SPY, QQQ) and total-market exposure (VTI), through sector plays (XLF, XLE, XLK, XLI, XLRE, EEM), to commodities (GLD) and fixed income (AGG) – and constructed two distinct strategies: conservative and aggressive. For each strategy, I imposed both per-asset and per-class caps which were designed to reflect realistic portfolio constraints and mandates, then used a Monte Carlo simulation and efficient-frontier analysis to identify the portfolio with the highest Sharpe ratio under each set of constraints. Finally, I benchmarked these constrained portfolios against unconstrained portfolios to highlight the effects of caps on risk-return trade-offs

## Dataset
Historical price data was downloaded via finance for the twelve ETFs over a ten-year time horizon, from January 1, 2015 to January 1, 2025. Adjusted closing prices were forward-filled to ensure its continuity, and any series missing more than 5% of its observations were dropped. The final dataset comprised daily returns for each ticker, which were then annualized (mean x 252 trading days) to compute expected returns, and used to construct a 12 x 12 annualized covariance matrix. No additional cleaning or imputation was required, as the finance output was already well-structured and cleaned. 

## Machine Learning and Optimization Methodology
1. _Return and Risk Estimation:_
   - Compute daily percentage returns, then annualize the mean vector and covariance matrix
2. _Constraint Specification:_
   - Asset-Level Cap: Each ETF weight <= 15% of the total portfolio
   - Class-Level Cap: For example, under the conservative strategy, large-cap equity exposure was <= 40%, fixed income <= 20%, commodities <= 15%, and so on. The aggressive strategy relaxed these class caps based on its risk appetite. The unconstrained case (all class caps = 100%) was also included for benchmark evaluation.
3. _Monte Carlo Simulation:_
    - For each strategy, draw 1,000 random weight vectors that satisfy both the asset_max and class_max constraints (which were enforced via simple rejection sampling). For each vector, calculate:
      - Annualized Portfolio Return: WTu
      - Annualized Portfolio Volatility: sqrt(wTΣw)
      - Sharpe Ratio:  wTu / sqrt(wTΣw)
4._ Efficient Frontier Construction:_
    - From the simulated portfolios, identify the weight combination with the maximum Sharpe ratio for each strategy. Aggregate the full set of (Return, Volatility, Sharpe) points into DataFrames for visualization
5. _Visualization:_
    - Plot the efficient frontier for all three cases (conservative, aggressive, unconstrained) on a single risk-return diagram
    - Generate a side-by-side bar chart comparing the optimal ETF weights under each strategy
    - Render pie charts showing the resulting allocation across asset classes (e.g. equities, fixed income, commodities, real estate) for each strategy

## Results
1. _Efficient Frontier Comparison:_
    - The **conservative** frontier lies in the middle, achieving annualized volatility around 14% with returns near 12%.
    - The **aggressive** frontier lies the most right, pushes the volatility to about 15% with returns between 13-14%.
    - The **unconstrained** frontier dominates the most left, it also achieves the lowest volatility again closer to 14%, with the highest returns also between 13-14%.
<img width="931" height="590" alt="Screenshot 2025-07-29 at 1 44 25 PM" src="https://github.com/user-attachments/assets/647f4b61-f2aa-470b-8633-a8f0f4829e15" />

2. _Optimal Weight Distributions:_
    - Under the **conservative** strategy, the optimizer leans heavily into AGG (fixed income) and GLD (commodity), with only modest allocations to equities (each were capped at 15%)
    - The **aggressive** strategy concentrates near the 15% asset maximum in higher-volatility sector ETFs (QQQ, EEM, XLK), with a lower weight in AGG
    - In the **unconstrained** case, the optimizer placed more weight in the highest-Sharpe ETFs, resulting in a noticeable tilt towards equity
<img width="1391" height="789" alt="Screenshot 2025-07-29 at 1 45 13 PM" src="https://github.com/user-attachments/assets/93c29403-8543-4676-be9d-36d5bc690f8d" />

3. _Asset-Class Allocation:_
    - The pie charts reveal that as the risk tolerance increases, between conservative and aggressive, the share of equities also climbs between each strategy.
<img width="598" height="521" alt="Screenshot 2025-07-29 at 1 44 38 PM" src="https://github.com/user-attachments/assets/d255f320-7f9c-4849-a63f-e5b8c6245f0b" />
<img width="616" height="427" alt="Screenshot 2025-07-29 at 1 44 51 PM" src="https://github.com/user-attachments/assets/e99318fa-2368-4037-b2a2-2b67cd611f9f" />
<img width="603" height="506" alt="Screenshot 2025-07-29 at 1 45 02 PM" src="https://github.com/user-attachments/assets/abf8004e-6131-411f-a67f-ded82b106336" />

## Lessons Learned
This project reinforced several key insights into portfolio construction:
1. _Constraint Effects:_
   - Imposing realistic asset and class caps meaningfully alters the frontier, illustrating the trade-off between unconstrained return maximization and the practical need for diversification and risk limits.
2._ Diversification Benefits:_
   - Even under aggressive mandates, holding a non-zero position in commodities and fixed income improved downside protection, emphasizing the classic “don’t put all your eggs in one basket” principle.
3. _Model Assumptions:_
   - The Monte Carlo approach assumes that the returns are multivariate normal and stationary, which may not hold in more volatile markets. Future iterations should incorporate a fat-tail risk measure or regime-switching model.
4. _Rebalancing and Transaction Costs:_
   - This static, single-period framework omits turnover and cost considerations. Extending the work to include transaction-cost-aware rebalancing schedules could yield more implementable strategies.
5. _Data History and Survivorship Bias:_
   - Using a ten-year window provides robustness but may inadvertently benefit surviving ETFs. Incorporating delisted ETFs or a longer history could offer greater insight. 

Overall, this project demonstrates how quantitative techniques rooted in Modern Portfolio Theory can be tailored to practical investment mandates, striking a balance between empirical risk-return optimization and real world constraints. Continuous refinements – through alternative risk metrics, dynamic rebalancing, and stress-testing – will further enhance the reliability and applicability of these optimized ETF portfolios. 
