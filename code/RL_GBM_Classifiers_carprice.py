
# The RL GBM Classifiers for Predicting Above-Median Used Car Prices Using Kaggle Dataset
# The binary target is price > median(price)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt


# === Step 1: Load Data: https://www.kaggle.com/competitions/playground-series-s4e9/data ===
# The binary target is price > median(price)
df = pd.read_csv("train.csv")

# === Step 2: Initial Cleanup ===
# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Drop rows where target (price) is missing or 0
df = df[df['price'].notnull() & (df['price'] > 0)]

# === Step 3: Clean and Extract Numeric Info ===
def extract_hp(text):
    match = re.search(r"(\d+\.?\d*)\s*HP", str(text))
    return float(match.group(1)) if match else np.nan

def extract_engine_size(text):
    match = re.search(r"(\d+\.\d+)L", str(text))
    return float(match.group(1)) if match else np.nan

def extract_cylinder_count(text):
    match = re.search(r"(\d+)\s*[Vv]?\s*[Cc]ylinder", str(text))
    return int(match.group(1)) if match else np.nan

df['engine_hp'] = df['engine'].apply(extract_hp)
df['engine_L'] = df['engine'].apply(extract_engine_size)
df['cylinder'] = df['engine'].apply(extract_cylinder_count)

# Drop original engine column
df.drop(columns=['engine'], inplace=True)

# === Step 4: Handle Missing Values ===
# Add missing flags for selected columns
for col in ['int_col', 'transmission']:
    df[f'flag_{col}_missing'] = df[col].isnull().astype(int)

# Fill missing with placeholder
df['int_col'] = df['int_col'].fillna('Missing')
df['transmission'] = df['transmission'].fillna('Missing')

# Fill numeric engine values
for col in ['engine_hp', 'engine_L', 'cylinder']:
    df[col] = df[col].fillna(df[col].median())

# === Step 5: Categorical One-hot Encoding ===
cat_cols = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# === Step 6: Drop unnecessary or ID fields ===
df.drop(columns=['id'], inplace=True)

# === Step 7: Correlation with Target (price) ===
target = 'price'
features = [col for col in df.columns if col != target]
corr_values = df[features].apply(lambda x: x.corr(df[target]))
abs_corr = corr_values.abs().sort_values(ascending=False)

# Select top N features
top_n = 20
top_features = abs_corr.head(top_n).index.tolist()

# === Step 8: Final Model Data ===
model_df = df[top_features + [target]].dropna()


#############################################
# Metric & Utility Functions
#############################################
def compute_ks(y_true, y_prob):
    pos = (y_true==1).sum()
    neg = (y_true==0).sum()
    if pos == 0 or neg == 0: return 0.0
    df_temp = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob}).sort_values('y_prob', ascending=False)
    df_temp['cum_pos'] = (df_temp['y_true'] == 1).cumsum() / pos
    df_temp['cum_neg'] = (df_temp['y_true'] == 0).cumsum() / neg
    return max(abs(df_temp['cum_pos'] - df_temp['cum_neg']))

def acc_from_proba(y_true, y_prob, threshold=0.5):
    return accuracy_score(y_true, (y_prob >= threshold).astype(int))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def get_grad_hess_logistic(y_true, f):
    p = sigmoid(f)
    grad = (y_true - p)
    hess = p * (1 - p)
    return grad, hess

#############################################
# RL-GBM Classifier
#############################################
def rl_gbm_classifier(X_train, y_train, X_val, y_val, X_test, y_test,
                      base_learn_rate=0.1, M=200, max_depth=3,
                      actions=[0.95, 1.0, 1.05]):
    n_train = len(X_train)
    feature_names = X_train.columns
    Q = np.zeros((10, len(actions)))
    epsilon = 0.1
    alpha_q = 0.1
    gamma_q = 0.9
    F_train = np.zeros(n_train)
    F_val = np.zeros(len(X_val))
    F_test = np.zeros(len(X_test))
    feature_importance_sum = np.zeros(len(feature_names), dtype=np.float64)

    def get_state(g): return min(int(np.mean(np.abs(g)) / 0.05), 9)
    def predict_prob(f): return sigmoid(f)

    old_val_auc = roc_auc_score(y_val, predict_prob(F_val))
    train_auc_hist, val_auc_hist, test_auc_hist = [], [], []

    for t in range(M):
        grad_train, hess_train = get_grad_hess_logistic(y_train, F_train)
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=1)
        tree.fit(X_train, grad_train, sample_weight=hess_train)
        g_pred_train = tree.predict(X_train)
        g_pred_val = tree.predict(X_val)
        g_pred_test = tree.predict(X_test)

        state = get_state(grad_train)
        action_idx = np.random.randint(len(actions)) if np.random.rand() < epsilon else np.argmax(Q[state])
        alpha_eff = base_learn_rate * actions[action_idx]

        F_train += alpha_eff * g_pred_train
        F_val += alpha_eff * g_pred_val
        F_test += alpha_eff * g_pred_test
        feature_importance_sum += alpha_eff * tree.feature_importances_

        new_val_auc = roc_auc_score(y_val, predict_prob(F_val))
        reward = new_val_auc - old_val_auc
        old_val_auc = new_val_auc
        next_state = get_state(y_train - sigmoid(F_train))
        Q[state, action_idx] += alpha_q * (reward + gamma_q * np.max(Q[next_state]) - Q[state, action_idx])

        train_auc_hist.append(roc_auc_score(y_train, predict_prob(F_train)))
        val_auc_hist.append(new_val_auc)
        test_auc_hist.append(roc_auc_score(y_test, predict_prob(F_test)))

    final_prob = sigmoid(F_test)
    if feature_importance_sum.sum() > 0:
        feature_importance_sum /= feature_importance_sum.sum()

    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'RLGBM_Importance': feature_importance_sum
    }).sort_values('RLGBM_Importance', ascending=False)

    print("\n=== RL-GBM Classifier: Feature Importances ===")
    print(feat_imp_df)
    return final_prob, train_auc_hist, val_auc_hist, test_auc_hist, feat_imp_df

def rl_gbm_classifier_ks(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    base_learn_rate=0.1,
    M=200,
    max_depth=3,
    actions=[0.95, 1.0, 1.05]
):
    """
    RL-GBM Classifier (logistic):
    RL agent chooses learning rate multiplier to maximize KS on validation set.
    """

    n_train = len(X_train)
    feature_names = X_train.columns

    # RL parameters
    num_states = 10
    num_actions = len(actions)
    Q = np.zeros((num_states, num_actions))
    epsilon = 0.1
    alpha_q = 0.1
    gamma_q = 0.9

    # Ensemble log-odds predictions
    F_train = np.zeros(n_train)
    F_val   = np.zeros(len(X_val))
    F_test  = np.zeros(len(X_test))

    # Feature importance accumulator
    feature_importance_sum = np.zeros(len(feature_names), dtype=np.float64)

    def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
    def get_state(g): return min(int(np.mean(np.abs(g)) / 0.05), num_states - 1)

    # Initial KS on validation
    old_val_prob = sigmoid(F_val)
    old_val_ks = compute_ks(y_val, old_val_prob)

    # History
    train_ks_hist, val_ks_hist, test_ks_hist = [], [], []

    for t in range(M):
        # 1. Compute gradient & hessian
        p_train = sigmoid(F_train)
        grad = y_train - p_train
        hess = p_train * (1 - p_train)

        # 2. Fit regression tree to gradient
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=1)
        tree.fit(X_train, grad, sample_weight=hess)

        # 3. Predict pseudo-residuals
        g_pred_train = tree.predict(X_train)
        g_pred_val   = tree.predict(X_val)
        g_pred_test  = tree.predict(X_test)

        # 4. RL picks action (learning rate multiplier)
        state = get_state(grad)
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(num_actions)
        else:
            action_idx = np.argmax(Q[state])
        alpha_eff = base_learn_rate * actions[action_idx]

        # 5. Update predictions
        F_train += alpha_eff * g_pred_train
        F_val   += alpha_eff * g_pred_val
        F_test  += alpha_eff * g_pred_test
        feature_importance_sum += alpha_eff * tree.feature_importances_

        # 6. Compute new validation KS (reward)
        new_val_prob = sigmoid(F_val)
        new_val_ks = compute_ks(y_val, new_val_prob)
        reward = new_val_ks - old_val_ks
        old_val_ks = new_val_ks

        # 7. Q update
        next_state = get_state(y_train - sigmoid(F_train))
        Q[state, action_idx] += alpha_q * (
            reward + gamma_q * np.max(Q[next_state]) - Q[state, action_idx]
        )

        # 8. Track history
        train_ks_hist.append(compute_ks(y_train, sigmoid(F_train)))
        val_ks_hist.append(new_val_ks)
        test_ks_hist.append(compute_ks(y_test, sigmoid(F_test)))

    # Final predictions
    test_prob = sigmoid(F_test)

    # Normalize and display feature importances
    total = feature_importance_sum.sum()
    if total > 0:
        feature_importance_sum /= total

    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'RLGBM_Importance': feature_importance_sum
    }).sort_values('RLGBM_Importance', ascending=False)

    print("\n=== RL-GBM Classifier (KS-Driven): Feature Importances ===")
    print(feat_imp_df)

    return test_prob, train_ks_hist, val_ks_hist, test_ks_hist, feat_imp_df

#############################################
# Load & Prepare Car Price Data
#############################################

df = model_df.copy()
df['target'] = (df['price'] > df['price'].median()).astype(int)


features = [
    'brand_Lamborghini', 'transmission_A/T', 'engine_L', 'cylinder',
    'brand_Bentley', 'brand_Porsche', 'int_col_Nero Ade',
    'transmission_7-Speed Automatic with Auto-Shift', 'transmission_6-Speed A/T',
    'transmission_8-Speed Automatic with Auto-Shift', 'int_col_Gray',
    'int_col_Beige', 'transmission_8-Speed Automatic', 'model_911 GT3',
    'brand_Rolls-Royce', 'transmission_8-Speed A/T'
]

X = df[features].copy()
y = df['target'].copy()

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, stratify=y_train_val, test_size=0.25, random_state=42)

#############################################
# Fit RL-GBM
#############################################

test_prob, train_auc_hist, val_auc_hist, test_auc_hist, feat_imp_df = rl_gbm_classifier(
    X_train, y_train, X_val, y_val, X_test, y_test,
    base_learn_rate=0.05, M=400, max_depth=5)

'''
test_prob, train_auc_hist, val_auc_hist, test_auc_hist, feat_imp_df = rl_gbm_classifier_ks(
    X_train, y_train, X_val, y_val, X_test, y_test,
    base_learn_rate=0.04, M=500, max_depth=4
)
'''

# Evaluate RL-GBM
auc_rl = roc_auc_score(y_test, test_prob)
ks_rl = compute_ks(y_test, test_prob)
acc_rl = accuracy_score(y_test, (test_prob >= 0.5).astype(int))


#############################################
# Benchmark Classifiers
#############################################
xgb_clf = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1,
                            objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)
xgb_prob = xgb_clf.predict_proba(X_test)[:, 1]

lgb_clf = lgb.LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.1)
lgb_clf.fit(X_train, y_train)
lgb_prob = lgb_clf.predict_proba(X_test)[:, 1]

ada_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=150, learning_rate=1.0)
ada_clf.fit(X_train, y_train)
ada_prob = ada_clf.predict_proba(X_test)[:, 1]

rf_clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
rf_clf.fit(X_train, y_train)
rf_prob = rf_clf.predict_proba(X_test)[:, 1]

print("\n=== RL-GBM Classifier ===")
print(f"AUC: {auc_rl:.4f}, KS: {ks_rl:.4f}, ACC: {acc_rl:.4f}")
print(f"XGBoost => AUC: {roc_auc_score(y_test,xgb_prob):.4f}, KS: {compute_ks(y_test,xgb_prob):.4f}, ACC: {accuracy_score(y_test,(xgb_prob>=0.5).astype(int)):.4f}")
print(f"AdaBoost => AUC: {roc_auc_score(y_test,ada_prob):.4f}, KS: {compute_ks(y_test,ada_prob):.4f}, ACC: {accuracy_score(y_test,(ada_prob>=0.5).astype(int)):.4f}")
print(f"RandomForest => AUC: {roc_auc_score(y_test,rf_prob):.4f}, KS: {compute_ks(y_test,rf_prob):.4f}, ACC: {accuracy_score(y_test,(rf_prob>=0.5).astype(int)):.4f}")
print(f"LightGBM => AUC: {roc_auc_score(y_test,lgb_prob):.4f}, KS: {compute_ks(y_test,lgb_prob):.4f}, ACC: {accuracy_score(y_test,(lgb_prob>=0.5).astype(int)):.4f}")

#############################################
# Plot AUC Over Iterations
#############################################
plt.figure(figsize=(8, 6))
plt.plot(train_auc_hist, label="Train AUC", linewidth=2)
plt.plot(val_auc_hist, label="Val AUC", linewidth=2)
plt.plot(test_auc_hist, label="Test AUC", linewidth=2)
plt.xlabel("Boosting Iteration")
plt.ylabel("AUC")
plt.title("RL-GBM Classifier AUC Over Iterations")
plt.legend()
plt.grid(True)
plt.show()


=== RL-GBM Classifier ===
AUC: 0.7749, KS: 0.4206, ACC: 0.7083
XGBoost => AUC: 0.7763, KS: 0.4272, ACC: 0.7125
AdaBoost => AUC: 0.7621, KS: 0.4091, ACC: 0.7040
RandomForest => AUC: 0.7669, KS: 0.4099, ACC: 0.6997
LightGBM => AUC: 0.7755, KS: 0.4257, ACC: 0.7107
