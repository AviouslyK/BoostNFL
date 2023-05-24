import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
from xgboost import plot_importance

# Read in public NFL dataset
X_full = pd.read_csv("http://www.habitatring.com/games.csv") 
X_test_full = pd.read_csv("http://www.habitatring.com/games.csv")

# Remove rows with missing target
X_full = X_full.dropna( how='any', subset=['away_score','home_score','total','result'])

# Separate out Training/Validation data
# Consider all games in data set prior to current week
#current_week = 5
X_full = X_full[X_full['game_type'] == "REG"] 
#X_full = X_full[ ((X_full['season'] == 2022) & (X_full['week'] <= current_week)) | X_full['season'] < 2022 ]
X_full = X_full[X_full['season'] < 2009]
# Separate out Testing Data
#X_test_full = X_test_full[(X_test_full['season'] >= 2022) & (X_test_full['week'] > current_week) & (X_test_full['game_type'] == "REG")] 
X_test_full = X_test_full[(X_test_full['season'] >= 2009) & (X_test_full['game_type'] == "REG")]
y_test = X_test_full.result 

# separate target from predictors
y = X_full.result # home_team points - visiting_team points 
X_full.drop(['result'], axis=1, inplace=True) # this is target
X_full.drop(['away_score'], axis=1, inplace=True) # these are directlly related to target
X_full.drop(['home_score'], axis=1, inplace=True)
X_full.drop(['total'], axis=1, inplace=True) 

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.7, test_size=0.3, random_state=0)

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

# Select categorical columns
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 1000 and 
                    X_train_full[cname].dtype == "object" or X_train_full[cname].dtype == "string"]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# Save copies of the data before one-hot encoding, for showing results later on
X_train_OG = X_train
X_valid_OG = X_valid
X_test_OG = X_test

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

X_train.name = 'Training Set'
X_test.name = 'Test Set'

print('Number of Training Examples = {}'.format(X_train.shape[0]))
print('Number of Validation Examples = {}'.format(X_valid.shape[0]))
print('Number of Test Examples = {}\n'.format(X_test.shape[0]))

print(my_cols)
print("\n")

# Define the model
model = XGBRegressor(n_estimators=100, learning_rate=0.025,  early_stopping_rounds=4, max_depth=3, subsample=0.85) 
# Fit the model
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
# Get predictions
preds = model.predict(X_valid)
print('MAE:', mean_absolute_error(y_valid, preds))
print('MSE:', mean_squared_error(y_valid, preds))

# Predict Test Data
df_total = pd.concat([y_test, X_test_OG], axis=1)
df_total["Pred"] = model.predict(X_test)
df_total.Pred = df_total.Pred.round()

df_total = df_total[ ['Pred'] + [ col for col in df_total.columns if col != 'Pred' ] ]
df_total.to_csv('boosted_prediction.csv')

# plot prediction vs actual 
df_total["Game"] = df_total.index
df_total["Model-Real"] = df_total["Pred"] - df_total["result"]
plot = df_total.plot(x="Game",y="Model-Real")
fig = plot.get_figure()
fig.savefig("boosted_output2.png")

# See how features relate to themselves and the target
g = sns.pairplot(df_total, vars=["div_game","result","Model-Real","season"], hue = "overtime")
g.fig.savefig("seaborn.png")
# Investigate Feature Importance
g = plot_importance(model)
fig = g.get_figure()
fig.savefig("feat_import.png")
plt.show()