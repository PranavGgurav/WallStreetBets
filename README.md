# WallStreetBets
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

#import s&p 500
data = pd.read_csv('all_stocks_5yr.csv')
data

date	open	high	low	close	volume	Name
0	2013-02-08	15.07	15.12	14.63	14.75	8407500	AAL
1	2013-02-11	14.89	15.01	14.26	14.46	8882000	AAL
2	2013-02-12	14.45	14.51	14.10	14.27	8126000	AAL
3	2013-02-13	14.30	14.94	14.25	14.66	10259500	AAL
4	2013-02-14	14.94	14.96	13.16	13.99	31879900	AAL
...	...	...	...	...	...	...	...
619035	2018-02-01	76.84	78.27	76.69	77.82	2982259	ZTS
619036	2018-02-02	77.53	78.12	76.73	76.78	2595187	ZTS
619037	2018-02-05	76.64	76.92	73.18	73.83	2962031	ZTS
619038	2018-02-06	72.74	74.56	72.13	73.27	4924323	ZTS
619039	2018-02-07	72.70	75.00	72.69	73.86	4534912	ZTS
619040 rows × 7 columns

data['close'].plot()

train_data = data.iloc[:int(.99*len(data)), :]
test_data = data.iloc[int(.99*len(data)):, :]

features = ['open', 'volume']
target = 'close'

model = xgb.XGBRegressor()
model.fit(train_data[features], train_data[target])


XGBRegressor
XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=None, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=None, n_jobs=None,
             num_parallel_tree=None, random_state=None, ...)


predictions = model.predict(test_data[features])
print('Model Predictions:')
print(predictions)

Model Predictions:
[28.696836 29.5231   29.134607 ... 76.75294  72.857925 72.857925]

print('Actual Values:')
print(test_data[target])


Actual Values:
612849    28.97
612850    29.03
612851    29.08
612852    29.06
612853    28.58
          ...  
619035    77.82
619036    76.78
619037    73.83
619038    73.27
619039    73.86
Name: close, Length: 6191, dtype: float64


accuracy = model.score(test_data[features], test_data[target])
print('Accuracy:')
print(accuracy)\

Accuracy:
0.999329708731666

plt.plot(predictions, label = 'predictions')
plt.legend()
plt.show()
