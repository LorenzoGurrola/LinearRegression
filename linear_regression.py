
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
sys.path.append(
    r'c:\users\lorenzo\desktop\worskspace\github\linearregression\.venv\lib\site-packages')
sys.path.append(
    r'c:\users\lorenzo\desktop\worskspace\github\linearregression\.venv\lib\site-packages')

data = pd.read_csv(
    'https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')
train, test = train_test_split(data, train_size=0.75, random_state=10)

x_train_raw = np.array(train['Experience Years'])
y_train_raw = np.array(train['Salary'])
x_test_raw = np.array(test['Experience Years'])
y_test_raw = np.array(test['Salary'])


def normalize(data):
    min = np.min(data)
    max = np.max(data)
    data_norm = (data - min)/(max-min)
    return data_norm


x_train = normalize(x_train_raw)

print('saasdfdf')
print('butt')
print('asdf')
