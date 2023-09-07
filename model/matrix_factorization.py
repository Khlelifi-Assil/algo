# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from MatrixFactorization import MatrixFactorization
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('input/dataset/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

customers=pd.read_json("input/dataset/customers.json")
print(customers.size)
print(customers)
products=pd.read_json("input/dataset/products.json")
print(products.size)
print(products)
ratings=pd.read_json("input/dataset/ratings.json")
print(ratings.size)
print(ratings)

# We don't care CreateDate attribute, so we remove this column for ratings
ratings.drop('CreateDate', inplace=True, axis=1)


rate_train =ratings[0:129000]
rate_train = np.array(rate_train)
rate_test = ratings[129001:]
rate_test= np.array(rate_test)
print('Number of traing rates:', rate_train.shape[0])
print('Number of test rates:', rate_test.shape[0])

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1
mf = MatrixFactorization(rate_train,customers,products,K = 50, lam = .01, print_every = 5, learning_rate = 50,max_iter = 30)
mf.fit()
# evaluate on test data
RMSE = mf.evaluate_RMSE(rate_test)
print("\nMatrix Factorization CF, RMSE = %.4f" %RMSE)

print(rate_test)

expected_score=3.8
print("Expected Score =",expected_score)
for c in customers.values[0:10]:
    customerId=c[0]
    customerName=c[1]
    print("Customer [",customerId,customerName,"], recommendation products:")
    for p in products.values:
        productId=p[0]
        productName=p[1]
        result=mf.predict(customerId,productId)
        if result>=expected_score:
            print("\t Recommend Product [",productName, "] Score=",result)