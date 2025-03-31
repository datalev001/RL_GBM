#######The RL GBM Regression for predicting Amount Using Using Synthetic Data####################
# The target of regression models is 'Amount'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

#############################################
# load Data
#############################################

# The target of regression models is 'Amount'
df = pd.read_csv('purchase.csv')
df = pd.get_dummies(df, columns=['Channel'], drop_first=True)

#############################################
# Metric Functions
#############################################
def mean_absolute_percentage_error(y_true, y_pred):
    eps = 1e-6
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)) * 100.0

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

#############################################
# RL-GBM Regressor
#############################################
def rl_gbm_regressor(X_train, y_train, X_val, y_val, X_test, y_test,
                     base_learn_rate=0.1, M=200,
                     max_depth=3,
                     actions=[0.8, 0.9, 1.0, 1.1, 1.2]):
    """
    RL-GBM Regressor:
      - Each iteration, we compute gradients g_i = y_i - F_{t-1}(x_i)
        (assuming MSE => negative gradient is g_i).
      - Fit a regression tree h_t on (X_train, g_i).
      - Some standard line search => we define a base LR = base_learn_rate.
      - RL picks a multiplier from the 'actions' set => alpha_eff = base_learn_rate * RL_multiplier.
      - Then F_{t}(x) = F_{t-1}(x) + alpha_eff * h_t(x).
      - Reward => improvement in validation MAPE. 
      - Also accumulates feature importances from the tree * alpha_eff each iteration.
    """
    # Dimensions
    n_train = len(X_train)
    n_features = X_train.shape[1]
    feature_names = X_train.columns
    
    # RL Q-learning table
    num_states = 10
    num_actions = len(actions)
    Q = np.zeros((num_states, num_actions))
    epsilon = 0.1
    alpha_q = 0.1
    gamma_q = 0.9
    
    # Ensemble predictions
    F_train = np.zeros(n_train)
    F_val   = np.zeros(len(X_val))
    F_test  = np.zeros(len(X_test))
    
    # For feature importance
    feature_importance_sum = np.zeros(n_features, dtype=np.float64)
    
    # We'll define a "state" from the average absolute gradient or similar
    def get_state(g):
        # average gradient in [0, ???], let's just discretize by .1 increments
        avg_g = np.mean(np.abs(g))
        idx = int(avg_g / 10.0)  # or something
        idx = min(idx, num_states-1)
        return idx
    
    # Initial val MAPE
    prev_val_mape = mean_absolute_percentage_error(y_val, F_val)
    
    # We track MAPE across iterations
    train_mape_hist = []
    val_mape_hist   = []
    test_mape_hist  = []
    
    for t in range(M):
        # Negative gradient (for MSE => y_i - F_train_i)
        g_train = y_train - F_train
        
        # Fit a regression tree to (X_train, g_train)
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=1)
        tree.fit(X_train, g_train)
        # Feature importance from this tree
        tree_importance = tree.feature_importances_
        
        # partial predictions of the gradient
        g_pred_train = tree.predict(X_train)
        g_pred_val   = tree.predict(X_val)
        g_pred_test  = tree.predict(X_test)
        
        # define base step = base_learn_rate
        # RL picks multiplier
        state = get_state(g_train)
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(num_actions)
        else:
            action_idx = np.argmax(Q[state])
        multiplier = actions[action_idx]
        
        alpha_eff = base_learn_rate * multiplier
        
        # accumulate feature importance
        feature_importance_sum += alpha_eff * tree_importance
        
        # update ensemble predictions
        F_train += alpha_eff * g_pred_train
        F_val   += alpha_eff * g_pred_val
        F_test  += alpha_eff * g_pred_test
        
        # compute new val MAPE => reward
        val_mape_now = mean_absolute_percentage_error(y_val, F_val)
        reward = (prev_val_mape - val_mape_now)  # improvement => +ve
        prev_val_mape = val_mape_now
        
        # update Q
        next_state = get_state(y_train - F_train)
        Q[state, action_idx] += alpha_q * (
            reward + gamma_q * np.max(Q[next_state]) - Q[state, action_idx]
        )
        
        # track MAPE
        train_mape = mean_absolute_percentage_error(y_train, F_train)
        test_mape  = mean_absolute_percentage_error(y_test,  F_test)
        
        train_mape_hist.append(train_mape)
        val_mape_hist.append(val_mape_now)
        test_mape_hist.append(test_mape)
    
    # create feature importance DataFrame
    total = feature_importance_sum.sum()
    if total > 0:
        feature_importance_sum /= total
    feat_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'RLGBM_Importance': feature_importance_sum
    }).sort_values('RLGBM_Importance', ascending=False).reset_index(drop=True)
    
    print("\n=== RL-GBM Feature Importances ===")
    print(feat_importance_df)
    
    return F_test, Q, train_mape_hist, val_mape_hist, test_mape_hist


train_val, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.25, random_state=42)

train = train.reset_index(drop=True)
val   = val.reset_index(drop=True)
test  = test.reset_index(drop=True)

features = [c for c in df.columns if c not in ['Amount','Promo']]
X_train = train[features]
y_train = train['Amount']
X_val   = val[features]
y_val   = val['Amount']
X_test  = test[features]
y_test  = test['Amount']

#############################################
# RL-GBM
#############################################
final_test_preds, Q_table, train_mape_hist, val_mape_hist, test_mape_hist = rl_gbm_regressor(
    X_train, y_train, X_val, y_val, X_test, y_test,
    base_learn_rate=0.1, M=200, max_depth=3
)

# Evaluate with MAPE, RMSE
mape_rl  = mean_absolute_percentage_error(y_test, final_test_preds)
rmse_rl  = rmse(y_test, final_test_preds)
print("\n=== RL-GBM Regressor Results ===")
print(f"MAPE(%): {mape_rl:.4f}, RMSE: {rmse_rl:.4f}")

#############################################
# Compare with XGB, LGB, RF, etc.
#############################################
xgb_reg = xgb.XGBRegressor(
    n_estimators=150, max_depth=3, learning_rate=0.1,
    objective='reg:squarederror', random_state=42
)
xgb_reg.fit(X_train, y_train)
xgb_preds = xgb_reg.predict(X_test)
xgb_mape  = mean_absolute_percentage_error(y_test, xgb_preds)
xgb_rmse_ = rmse(y_test, xgb_preds)

lgb_reg = lgb.LGBMRegressor(
    n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42
)
lgb_reg.fit(X_train, y_train)
lgb_preds = lgb_reg.predict(X_test)
lgb_mape  = mean_absolute_percentage_error(y_test, lgb_preds)
lgb_rmse_ = rmse(y_test, lgb_preds)

rf_reg = RandomForestRegressor(
    n_estimators=150, max_depth=10, random_state=42
)
rf_reg.fit(X_train, y_train)
rf_preds = rf_reg.predict(X_test)
rf_mape  = mean_absolute_percentage_error(y_test, rf_preds)
rf_rmse_ = rmse(y_test, rf_preds)

print("\n=== Performance Comparison ===")
print(f"RL-GBM => MAPE: {mape_rl:.4f}  RMSE: {rmse_rl:.4f}")
print(f"XGBoost => MAPE: {xgb_mape:.4f}  RMSE: {xgb_rmse_:.4f}")
print(f"LightGBM => MAPE: {lgb_mape:.4f}  RMSE: {lgb_rmse_:.4f}")
print(f"RandomForest => MAPE: {rf_mape:.4f}  RMSE: {rf_rmse_:.4f}")

# Plot MAPE histories
plt.figure(figsize=(8,6))
plt.plot(train_mape_hist, label="Train MAPE", linewidth=2)
plt.plot(val_mape_hist,   label="Val MAPE", linewidth=2)
plt.plot(test_mape_hist,  label="Test MAPE", linewidth=2)
plt.xlabel("Boosting Iteration")
plt.ylabel("MAPE (%)")
plt.title("RL-GBM Regressor MAPE Over Iterations")
plt.legend()
plt.show()


#######result of RL GBM regression##############

=== RL-GBM Feature Importances ===
          Feature  RLGBM_Importance
0          Income          0.485857
1             Age          0.294284
2            Days          0.139355
3         Holiday          0.079364
4         Loyalty          0.001118
5  Channel_Mobile          0.000021
6  Channel_Online          0.000000

=== RL-GBM Regressor Results ===
MAPE(%): 1.7919, RMSE: 103.4443

=== Performance Comparison ===
RL-GBM => MAPE: 1.7919  RMSE: 103.4443
XGBoost => MAPE: 2.1236  RMSE: 147.1589
LightGBM => MAPE: 2.0508  RMSE: 143.5815
RandomForest => MAPE: 2.1796  RMSE: 154.4802
