import resources
import datetime as dt
from pathlib import Path
from resources import Order, load_account, save_account,\
    OPERATION_BUY, OPERATION_SELL
from account_manager import rebalance_account
path = Path(__file__)
resources.accounts_path = path.parent.parent / 'accounts/'

if __name__ == '__main__':
    
    account = load_account('Charlie')
    
    # Rebalance the porfolio and shows the new portfolio of the orders
    # are (manually) executed.
    new_orders = rebalance_account(account,
                                   additional_cash=10_000,
                                   start_date=dt.datetime(2020, 1, 1),
                                   end_date=dt.datetime(2021, 12, 17),
                                   cvar_beta=0.90,
                                   ignored_securities=[
                                       'GME',
                                   ])
    
    print(new_orders)
