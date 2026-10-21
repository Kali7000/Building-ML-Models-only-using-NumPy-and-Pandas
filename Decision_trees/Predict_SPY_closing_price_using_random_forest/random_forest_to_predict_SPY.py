# -*- coding: utf-8 -*-
"""
@author: Kali
"""

import yfinance as yf
import pandas as pd
import numpy as np


stock_df = yf.download("SPY", start="2010-01-01", auto_adjust=True)

# print(stock_df.head())
# print(stock_df.tail())
# print(stock_df.info())