import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier

import pickle
import time
import os

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore") 


def train_xgboost(X_train, y_train, X_eval, y_eval, seed, method_name=None, save_dir='./', filename='xgb.pkl', verbose=1, labels=None, n_jobs=2):
    # train XGBoost
    clf = XGBClassifier(random_state=seed, n_jobs=n_jobs)
    clf.fit(X_train, y_train)

    # predictions
    y_pred = clf.predict(X_eval)

    # metrics
    acc = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred, average='macro')
    recall = recall_score(y_eval, y_pred, average='macro')
    f1 = f1_score(y_eval, y_pred, average='macro')

    if verbose > 0:
        # generate and display the confusion matrix
        cm = confusion_matrix(y_eval, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels.values()))
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(f'Confusion matrix: {method_name} - Seed {seed}')
        plt.savefig(f"../data/results/confusion_matrix/cm_{method_name.lower().replace(' ', '-')}_seed{seed}.png", bbox_inches='tight')
        plt.close()

        # feature importance
        df_importance = pd.DataFrame(
            zip(clf.get_booster().feature_names, clf.feature_importances_), 
            columns=['feature', 'importance']
        ).sort_values(by=['importance'], ascending=False)
        if df_importance.shape[0] > 30:  # for visibility
            df_importance = df_importance.iloc[:30]
        sns.barplot(df_importance, x='importance', y='feature')
        plt.title(f'XGBoost feature importance (gain) - {method_name} - Seed {seed}')
        plt.savefig(f"../data/results/feature_importance/fi_{method_name.lower().replace(' ', '-')}_seed{seed}.png", bbox_inches='tight')
        plt.close()

        print('')
        print('\tEvaluation accuracy:', acc)
        print('\tEvaluation precision:', precision)
        print('\tEvaluation recall:', recall)
        print('\tEvaluation f1-score:', f1)

    # save model to disk
    pickle.dump(clf, open(os.path.join(save_dir, filename), "wb"))
    
    return pd.DataFrame({
        'model': [method_name],
        'accuracy': [acc],
        'precision': [precision],
        'recall': [recall],
        'f1-score': [f1],
    })