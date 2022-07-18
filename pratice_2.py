import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.display.float_format = "{:,.2f}".format
pd.options.display.max_rows = 100
pd.options.display.max_columns = 20

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from scipy import stats

