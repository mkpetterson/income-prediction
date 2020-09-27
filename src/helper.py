import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def chi_squared_test(df: pd.DataFrame, label1: str, label2: str) -> int:
    """
    Chi-squraed test of independence
    Returns p-value. If p-value < sig, reject null that features independent
    """
    # Construct crosstab table with counts
    table = pd.crosstab(index=df[label1], columns=data[label2])
    stat, p, dof, expected = chi2_contingency(table.values)    
    return p

def impute_countries(df: pd.DataFrame) -> pd.DataFrame:
    """ Trying to be a little smart here. """
    countries = ['country_mother', 'country_father', 'country_self']
    
    # Replace self with country of mother
    mask = (df['country_self'] == ' ?') & (df['country_mother'] != ' ?')    
    df.loc[mask, 'country_self'] = df.loc[mask, 'country_mother']
    
    # Else replace self with country of father
    mask = (df['country_self'] == ' ?') & (df['country_mother'] == ' ?') & (df['country_father'] != ' ?')   
    df.loc[mask, 'country_self'] = df.loc[mask, 'country_father']
    
    # Fill self, mother, and father with most common value
    for c in countries:
        mode = df[c].value_counts().index[0]
        mask = df[c] == ' ?'
        df.loc[mask, c] = mode
        
    return df    
    
def preprocess(X: pd.DataFrame, cat_cols:list, num_cols:list, ohe=False):
    """ 
    OHE and StandardScaler for data using column transformer
    Normally would use df.select_dtypes() to get numerical vs categorical,
    but in this case some of the categorical features have label encoding.
    Inputs: dataframe, categorical labels, numerical labels, ohe encoding
    Returns: fit transformer and feature names
    """
    # Get index of cat and num columns
    all_cols = X.columns.to_list()
    num_cols_idx, cat_cols_idx = [], []
    for col in num_cols:
        num_cols_idx.append(all_cols.index(col))
    for col in cat_cols:
        cat_cols_idx.append(all_cols.index(col))
        
    # Implement OHE for logistic regression
    if ohe:
        t = [('cat', OneHotEncoder(), cat_cols_idx),
             ('num', StandardScaler(), num_cols_idx)]
        transformer = ColumnTransformer(transformers=t)
        transformer.fit(X)
        cat_names = list(transformer.named_transformers_['cat'].get_feature_names())
        feature_names =  cat_names + num_cols
    # Implement Ordinal for tree-based models
    else:
        t = [('cat', OrdinalEncoder(), cat_cols_idx),
             ('num', StandardScaler(), num_cols_idx)]
        transformer = ColumnTransformer(transformers=t)
        transformer.fit(X)  
        feature_names = cat_cols + num_cols
    
    return transformer, feature_names


def plot_roc(X: np.array, y: np.array, model, name: str) -> float:
    # Create figure
    tot_auc = 0
    fig, ax = plt.subplots()
    # Instantiate k-fold
    n_splits=5
    cv = StratifiedKFold(n_splits=n_splits)
    
    for n, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        
        tpr, fpr, thres = roc_curve(y_test, y_proba[:,1])
        auc = roc_auc_score(y_test, y_proba[:,1])
        tot_auc += auc
        label = 'KFold Split: {}, AUC Score: {:2.2f}'.format(n, auc)
        ax.plot(tpr, fpr, label=label)
        
    ax.plot([0,1],[0,1], '--', color='gray')
    ax.legend(loc='lower right')
    ax.set_title(f'ROC: {name}')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    plt.savefig('../images/roc.png', bbox_inches='tight', dpi=350)
    plt.show()
    
    return tot_auc/n_splits
    
    
def compile_metrics(X_train, X_test, y_train, y_test, model, log=False):
    """ Compiles metrics from model 
    Inputs: Train and test data, model
    Returns: metrics
    """
    
    # Get predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Get feature importances
    if log:
        importances = model.coef_[0]
    else:
        importances = model.feature_importances_
    
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
    plt.savefig('../images/cm.png', bbox_inches='tight', dpi=350)
    plt.show()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = f1_score(y_test, y_pred)
    
    return precision, recall, f1, importances


def feature_importances(feature_names, importances, name):
    features = pd.DataFrame({'Feature names': feature_names, 
                             'Feature Importances': importances}).sort_values(by='Feature Importances', 
                                                                                  ascending=True)
    top_10 = features.iloc[:10]
    bottom_10 = features.iloc[-10:]
    pd.concat([top_10, bottom_10]).set_index('Feature names').plot(kind='barh')
    plt.title('Top 10 and Bottom 10 feature importances')
    plt.savefig(f'../images/feature_import_{name}.png', bbox_inches='tight', dpi=250)
    plt.show()
    
    return None
    