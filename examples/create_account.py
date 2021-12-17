from resources import Order, create_new_account, OPERATION_BUY, save_account
import datetime as dt

if __name__ == '__main__':
    opening_date = dt.datetime(2021, 12, 12)
    account = create_new_account('Charlie', opening_date)
    account.deposit(opening_date, 1000)
    account.update_account(dt.datetime(2021, 12, 15, 14, 0), [
        Order('TSLA', 1, 952.99, OPERATION_BUY),
        Order('KR', 1, 44.79, OPERATION_BUY),
    ])
    save_account(account)
    print(account)
