import quandl
import numpy as np
import pandas as pd

mydata = quandl.get("WIKI/AAPL", start_date="1979-12-31", end_date="2016-12-31")
mydata.to_csv("/Users/tabish/MLProject/AAPL_stock_data.csv")
