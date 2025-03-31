import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor  # We'll fit gradient -> tree
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt

#############################################
# Metrics and Tools
#############################################
def compute_ks(y_true, y_prob):
    """
    Compute KS for binary classifier with predicted probs.
    """
    pos = (y_true==1).sum()
    neg = (y_true==0).sum()
    if pos==0 or neg==0: 
        return 0.0
    df_temp = pd.DataFrame({'y_true':y_true, 'y_prob':y_prob})
    df_temp = df_temp.sort_values('y_prob', ascending=False)
    df_temp['cum_pos'] = (df_temp['y_true']==1).cumsum()/pos
    df_temp['cum_neg'] = (df_temp['y_true']==0).cumsum()/neg
    return np.max(np.abs(df_temp['cum_pos'] - df_temp['cum_neg']))

def acc_from_proba(y_true, y_prob, threshold=0.5):
    return accuracy_score(y_true, (y_prob>=threshold).astype(int))

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

#############################################
# Grad & Hess for Logistic Loss
#############################################
def get_grad_hess_logistic(y_true, f):
    """
    y_true in {0,1}, logistic loss => negative gradient:
       grad_i = (y_i - p_i)
       hess_i = p_i*(1 - p_i)
    We'll produce 'grad' and 'hess' arrays for each sample.
    f is the current ensemble log-odds predictions.
    """
    p = sigmoid(f)
    grad = (y_true - p)
    hess = p*(1 - p)
    return grad, hess

#############################################
# RL-GBM Classifier
#############################################
def rl_gbm_classifier(
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
      - Each iteration: compute (grad, hess),
        fit a small regression tree to grad with sample_weight=hess.
      - RL picks multiplier from 'actions'.
      - Update F(x) = F(x) + alpha_eff*g_pred.
      - Feature importances accumulate => feature_importance_sum += alpha_eff*tree_importances
      - Reward => improvement in validation AUC.

    Returns test_prob, error histories, feature_importance_df
    """
    n_train = len(X_train)
    feature_names = X_train.columns
    
    # RL Q table
    num_states = 10
    num_actions = len(actions)
    Q = np.zeros((num_states, num_actions))
    epsilon = 0.1
    alpha_q = 0.1
    gamma_q = 0.9
    
    # Ensemble log-odds
    F_train = np.zeros(n_train)
    F_val   = np.zeros(len(X_val))
    F_test  = np.zeros(len(X_test))
    
    # For feature importance
    feature_importance_sum = np.zeros(len(feature_names), dtype=np.float64)
    
    # define a function that returns state from average abs grad
    def get_state(g):
        avg_g = np.mean(np.abs(g))
        idx = int(avg_g / 0.05)  # naive
        idx = min(idx, num_states-1)
        return idx
    
    # reward reference => define oldValAUC
    def predict_prob(f):
        return sigmoid(f)
    old_val_prob = predict_prob(F_val)
    old_val_auc  = roc_auc_score(y_val, old_val_prob)
    
    # track AUC over iterations
    train_auc_hist = []
    val_auc_hist   = []
    test_auc_hist  = []
    
    for t in range(M):
        # 1) Compute grad,hess for logistic
        grad_train, hess_train = get_grad_hess_logistic(y_train, F_train)
        
        # 2) Fit a regression tree to (X_train, grad), sample_weight=hess
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=1)
        tree.fit(X_train, grad_train, sample_weight=hess_train)
        
        # partial predictions of gradient
        g_pred_train = tree.predict(X_train)
        g_pred_val   = tree.predict(X_val)
        g_pred_test  = tree.predict(X_test)
        
        # 3) RL picks multiplier
        state = get_state(grad_train)
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(num_actions)
        else:
            action_idx = np.argmax(Q[state])
        multiplier = actions[action_idx]
        
        alpha_eff = base_learn_rate * multiplier
        
        # accumulate feature importances
        fi = tree.feature_importances_
        feature_importance_sum += alpha_eff*fi
        
        # 4) update ensemble
        F_train += alpha_eff*g_pred_train
        F_val   += alpha_eff*g_pred_val
        F_test  += alpha_eff*g_pred_test
        
        # measure newValAUC => reward
        new_val_prob = predict_prob(F_val)
        new_val_auc  = roc_auc_score(y_val, new_val_prob)
        reward = (new_val_auc - old_val_auc)
        old_val_auc = new_val_auc
        
        # update Q
        next_state = get_state(y_train - sigmoid(F_train))
        Q[state, action_idx] += alpha_q * (
            reward + gamma_q*np.max(Q[next_state]) - Q[state, action_idx]
        )
        
        # track train/val/test AUC
        tr_auc = roc_auc_score(y_train, predict_prob(F_train))
        te_auc = roc_auc_score(y_test,  predict_prob(F_test))
        
        train_auc_hist.append(tr_auc)
        val_auc_hist.append(new_val_auc)
        test_auc_hist.append(te_auc)
    
    # final predicted prob on test set
    test_prob = sigmoid(F_test)
    
    # create feature importance DataFrame
    total = feature_importance_sum.sum()
    if total>0:
        feature_importance_sum/= total
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'RLGBM_Importance': feature_importance_sum
    }).sort_values('RLGBM_Importance', ascending=False).reset_index(drop=True)
    
    print("\n=== RL-GBM Classifier: Weighted Feature Importances ===")
    print(feat_imp_df)
    
    return test_prob, train_auc_hist, val_auc_hist, test_auc_hist, feat_imp_df



if __name__=="__main__":
    # Generate data
    df = pd.read_csv('purchase_binary.csv')    
    df = df[df["Promo"]==1]
    df = pd.get_dummies(df, columns=['Channel'], drop_first=True)
    
    # Split
    from sklearn.model_selection import train_test_split
    train_val, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Purchase'])
    train, val = train_test_split(train_val, test_size=0.25, random_state=42, stratify=train_val['Purchase'])
    
    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)
    
    features= [c for c in df.columns if c not in ['Purchase','Promo']]
    X_train= train[features]
    y_train= train['Purchase']
    X_val=   val[features]
    y_val=   val['Purchase']
    X_test=  test[features]
    y_test=  test['Purchase']
    
    # Fit RL-GBM
    test_prob, train_auc_hist, val_auc_hist, test_auc_hist, feat_imp_df = rl_gbm_classifier(
        X_train, y_train, X_val, y_val, X_test, y_test,
        base_learn_rate=0.1,
        M=200,
        max_depth=3,
        actions=[0.95, 1.0, 1.05]
    )
    
    # Evaluate RL-GBM
    def compute_ks(y_true, y_prob):
        pos = (y_true==1).sum()
        neg = (y_true==0).sum()
        if pos==0 or neg==0: return 0.0
        df_temp=pd.DataFrame({'y_true':y_true,'y_prob':y_prob}).sort_values('y_prob',ascending=False)
        df_temp['cum_pos']=(df_temp['y_true']==1).cumsum()/pos
        df_temp['cum_neg']=(df_temp['y_true']==0).cumsum()/neg
        return max(abs(df_temp['cum_pos']-df_temp['cum_neg']))
    
    auc_rl= roc_auc_score(y_test, test_prob)
    acc_rl= accuracy_score(y_test, (test_prob>=0.5).astype(int))
    ks_rl= compute_ks(y_test, test_prob)
    
    print("\n=== RL-GBM Classifier (Logistic) ===")
    print(f"AUC: {auc_rl:.4f}, KS: {ks_rl:.4f}, ACC: {acc_rl:.4f}")
    
    # Compare with XGB, LGB, Ada, RF
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.tree import DecisionTreeClassifier
    
    # XGB
    xgb_clf= xgb.XGBClassifier(n_estimators=150,max_depth=3,learning_rate=0.1,
        objective='binary:logistic',use_label_encoder=False,eval_metric='logloss',random_state=42)
    xgb_clf.fit(X_train,y_train)
    xgb_prob= xgb_clf.predict_proba(X_test)[:,1]
    auc_xgb= roc_auc_score(y_test,xgb_prob)
    acc_xgb= accuracy_score(y_test,(xgb_prob>=0.5).astype(int))
    ks_xgb= compute_ks(y_test,xgb_prob)
    
    # LGB
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    lgb_clf.fit(X_train,y_train)
    lgb_prob= lgb_clf.predict_proba(X_test)[:,1]
    auc_lgb= roc_auc_score(y_test,lgb_prob)
    acc_lgb= accuracy_score(y_test,(lgb_prob>=0.5).astype(int))
    ks_lgb= compute_ks(y_test,lgb_prob)
    
    # AdaBoost
    ada_clf= AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=150,
        learning_rate=1.0,
        random_state=42
    )
    ada_clf.fit(X_train,y_train)
    ada_prob= ada_clf.predict_proba(X_test)[:,1]
    auc_ada= roc_auc_score(y_test,ada_prob)
    acc_ada= accuracy_score(y_test,(ada_prob>=0.5).astype(int))
    ks_ada= compute_ks(y_test,ada_prob)
    
    # RF
    rf_clf= RandomForestClassifier(n_estimators=150,max_depth=10,random_state=42)
    rf_clf.fit(X_train,y_train)
    rf_prob= rf_clf.predict_proba(X_test)[:,1]
    auc_rf= roc_auc_score(y_test,rf_prob)
    acc_rf= accuracy_score(y_test,(rf_prob>=0.5).astype(int))
    ks_rf= compute_ks(y_test,rf_prob)
    
    print("\n=== Performance Comparison ===")
    print(f"RL-GBM => AUC: {auc_rl:.4f}, KS: {ks_rl:.4f}, ACC: {acc_rl:.4f}")
    print(f"XGBoost => AUC: {auc_xgb:.4f}, KS: {ks_xgb:.4f}, ACC: {acc_xgb:.4f}")
    print(f"LightGBM => AUC: {auc_lgb:.4f}, KS: {ks_lgb:.4f}, ACC: {acc_lgb:.4f}")
    print(f"AdaBoost => AUC: {auc_ada:.4f}, KS: {ks_ada:.4f}, ACC: {acc_ada:.4f}")
    print(f"RandomForest => AUC: {auc_rf:.4f}, KS: {ks_rf:.4f}, ACC: {acc_rf:.4f}")
    
    # Plot AUC histories
    plt.figure(figsize=(8,6))
    plt.plot(train_auc_hist,label="Train AUC",linewidth=2)
    plt.plot(val_auc_hist,  label="Val AUC",linewidth=2)
    plt.plot(test_auc_hist, label="Test AUC",linewidth=2)
    plt.xlabel("Boosting Iteration")
    plt.ylabel("AUC")
    plt.title("RL-GBM Classifier (Logistic) AUC Over Iterations")
    plt.legend()
    plt.show()


##########RL GBM classifier#############
=== Performance Comparison ===
RL-GBM => AUC: 0.7523, KS: 0.3686, ACC: 0.6886
XGBoost => AUC: 0.7512, KS: 0.3689, ACC: 0.6898
LightGBM => AUC: 0.7510, KS: 0.3667, ACC: 0.6883
AdaBoost => AUC: 0.7506, KS: 0.3709, ACC: 0.6880
RandomForest => AUC: 0.7492, KS: 0.3653, ACC: 0.6856
