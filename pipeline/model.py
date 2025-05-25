import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier

import pickle
import time
import os


def train_xgboost(X_train, y_train, X_eval, y_eval, seed, method_name=None, save_dir='./', filename='xgb.pkl', verbose=1):
    # train XGBoost
    clf = XGBClassifier(random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)

    # predictions
    y_pred = clf.predict(X_eval)

    # generate and display the confusion matrix
    # cm = confusion_matrix(y_eval, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels.values()))
    # disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    # plt.title(f'Confusion matrix: {method_name}')
    # plt.show()

    # metrics
    acc = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred, average='macro')
    recall = recall_score(y_eval, y_pred, average='macro')
    f1 = f1_score(y_eval, y_pred, average='macro')
    
    # feature importance
    # df_importance = pd.DataFrame(
    #     zip(clf.get_booster().feature_names, clf.feature_importances_), 
    #     columns=['feature', 'importance']
    # ).sort_values(by=['importance'], ascending=False)
    # if df_importance.shape[0] > 20:  # for visibility
    #     df_importance = df_importance.iloc[:20]

    if verbose > 0:
        print('Evaluation accuracy:', acc)
        print('Evaluation precision:', precision)
        print('Evaluation recall:', recall)
        print('Evaluation f1-score:', f1)
    
    # sns.barplot(df_importance, x='importance', y='feature')
    # plt.title(f'XGBoost feature importance (gain) - {method_name}')
    # plt.show()

    # save model to disk
    pickle.dump(clf, open(os.path.join(save_dir, filename), "wb"))
    
    return pd.DataFrame({
        'model': [method_name],
        'accuracy': [acc],
        'precision': [precision],
        'recall': [recall],
        'f1-score': [f1],
    })