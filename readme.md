## <span style="color:red"> Disclaimer: NOT FINANCIAL ADVICE </span>

**This software is for informational and educational purposes only and does not constitute or intend to be financial advice. Investing is risky, do your own research and consult a financial advisor for the puropose of making and investment decision or otherwise.**

# Portfolio Manager

This repo contains code to download financial data using [ranaroussi/yfiance](https://github.com/ranaroussi/yfinance) and build a portfolio using mathematical optimization.

## Setup

Set your  `PYTHONPATH` to:

```shell
PYTHONPATH=path/to/PortfolioManager/source
```

## Create an account

To keep track of transactions, an [Account](https://github.com/dukduque/PortfolioManager/blob/3f014afdb6701846bf7c75cc72513414dfe55675/source/resources.py#L110) consolidates all the information associated to the history of a portfolio.

```python
from resources import Order, create_new_account, OPERATION_BUY, save_account
import datetime as 

opening_date = dt.datetime(2021, 12, 12)
account = create_new_account('name', opening_date)
account.deposit(opening_date, 1000)
account.update_account(dt.datetime(2021, 12, 15, 14, 0), [
    Order('TSLA', 1, 952.99, OPERATION_BUY),
    Order('KR', 1, 44.79, OPERATION_BUY),
])
save_account(account)
```

## Reblanace the portfolio

To rebalance the current portfolio, first load the account previously created and use the `rebalance_account` function from the `account_manager.py` module. This will indicate the orders to (manually) execute.

```python
import resources
import datetime as dt
from pathlib import Path
from resources import Order, load_account, save_account,\
    OPERATION_BUY, OPERATION_SELL
from account_manager import rebalance_account
path = Path(__file__)
resources.accounts_path = path.parent.parent / 'accounts/'

if __name__ == '__main__':
    account = load_account('name')
    
    # Rebalance the porfolio and shows the new portfolio of the orders
    # that are (manually) executed.
    new_orders = rebalance_account(account,
                                   additional_cash=10_000,
                                   start_date=dt.datetime(2020, 1, 1),
                                   end_date=dt.datetime(2021, 12, 17),
                                   cvar_beta=0.90,
                                   ignored_securities=[])
```

## Update data

The price data needs to be updated to rebalance the portfolio in the future. To do so, simply run:

```shell
python "path/to/database_handler.py" -a=u -db_file="close.pkl" -days_back=3 -n_proc=4
```
