# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
from sklearn.model_selection import ( train_test_split, KFold, StratifiedKFold,cross_val_score, RepeatedKFold, RandomizedSearchCV,learning_curve, ShuffleSplit, GridSearchCV)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import (IterativeImputer, SimpleImputer, KNNImputer)
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ( confusion_matrix , ConfusionMatrixDisplay, balanced_accuracy_score, roc_auc_score)
import plotly.graph_objects as go
import re
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMClassifier
import matplotlib.pylab as plt
np.seterr(divide = 'ignore')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
class SelectColumnsTransformer():
    #The parameter "columns" is set and only those columns will be resulted 
    def __init__(self, columns=None):
        self.columns = columns
        
    #Transformer function will copy only those columns which has better corelation with output variable
    def transform(self, X, **transform_params):
        cpy_df = X[self.columns].copy()
        return cpy_df

    def fit(self, X, y=None, **fit_params):
        return self

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
class DataframeFunctionTransformer():
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
class ml_support():
    def __init__(self):
        self.df = pd.read_csv("data/credit_risk_dataset.csv", encoding='latin')
        self.output_var = 'loan_status'
        self.num_cols = ['person_age','person_income','loan_amnt','loan_percent_income','cb_person_cred_hist_length']
        self.cat_cols = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file','higher_salary','home_owner','long_working','lower_loan_requirement','higher_loan_requirement']
        # These are columns without feature engineering
        self.cols_wofeature = [col for col in self.df.columns if col != self.output_var ]
        # These are coluns after feature engineering
        self.cols_wfeature = ['person_age','person_income','loan_amnt','loan_percent_income','cb_person_cred_hist_length','person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file','higher_salary','home_owner','long_working','lower_loan_requirement','higher_loan_requirement']
        self.X = self.df[self.cols_wofeature]
        #Output variable
        self.y = self.df[self.output_var]
        self.random_state = 42
    
    def drop_duplicate(self):
        
        self.df.drop_duplicates(inplace=True)
        return self.df
    
    def get_train_test_data(self):
        
        #Split traininig and testing set
        X_train, X_test , y_train, y_test = train_test_split(self.X,self.y,test_size=0.2,
                                                             random_state = self.random_state, shuffle=True, stratify=self.df[self.output_var])
        return X_train, X_test , y_train, y_test
    
    def get_column_transformer(self,is_estimator, iterative_Estimator, is_TreeBased=False):
        
        #Pipineline created for munging numerical columns
        num_pipeline_steps = []
        iterative_imputer = IterativeImputer()
        if is_estimator:
            iterative_imputer = IterativeImputer(estimator=iterative_Estimator)
        
        num_pipeline_steps.append(('num_missing',iterative_imputer))
        
        if not is_TreeBased:
            num_pipeline_steps.append(('num_smoothening',PowerTransformer()))
        
        num_pipeline = Pipeline(num_pipeline_steps)
        
        #Pipineline created for categorical column encoding
        cat_pipeline = Pipeline([
            ('cat_encoding',OneHotEncoder(sparse=False,drop='if_binary',handle_unknown='ignore'))
        ])
        
        # Column Transformer is created which will call pipelines
        ct = ColumnTransformer([
            ('num_munging',num_pipeline,self.num_cols),
            ('cat_munging',cat_pipeline,self.cat_cols)
        ])
        return ct
    
    def get_final_pipeline(self,regression_model,columntransformer, feature_engieered):
    
        X_train, X_test , y_train, y_test  = self.get_train_test_data()
        features_pipeline = [feature_engieered]
        features_pipeline.append(("selector", SelectColumnsTransformer(self.cols_wfeature)))
        features_pipeline.append(('munging',columntransformer))
        features_pipeline.append(('model',regression_model))
        finalized_pipeline = Pipeline(features_pipeline)
        return finalized_pipeline;
    
    def predit_with_pipeline(self,pipeline,X_train,X_test,y_train):
        pipeline.fit(X_train,y_train)
        preds = pipeline.predict(X_test)
        return preds
    
    def get_best_params(self,pipeline,params):
    
        rscv = RandomizedSearchCV(pipeline, params, scoring='balanced_accuracy',
                              n_jobs=-1, n_iter=4, cv=5, random_state=self.random_state, verbose=3)
        rscv.fit(self.X,self.y)
        return rscv.best_params_

    def get_best_params_gcv(self,pipeline,params):
    
        rscv = GridSearchCV(pipeline, params, scoring='balanced_accuracy',
                              n_jobs=-1, cv=5, verbose=3)
        rscv.fit(self.X,self.y)
        return rscv.best_params_
    
    def plot_confusion_matrix(self,y_test, preds, finalized_pipeline):
    
        cm = confusion_matrix(y_test, preds, labels=finalized_pipeline.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels= finalized_pipeline.classes_)
        disp.plot();
        plt.show();
        
    def update_performance_matrics(self,model_info,df):
        
        filter_con =  df["Model_Name"] == model_info['Model_Name']
        if ((filter_con)).any():
            df.loc[filter_con, 'Score'] = model_info['Score']
        else:
            df = df.append(model_info, ignore_index=True)
        return df
    
    def plot_learning_curves(self,estimator):
        """
        Don't forget to change the scoring and plot labels
        based on the metric that you are using.
        """
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=estimator,
            X=self.X,
            y=self.y,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=5,
            scoring="balanced_accuracy",
            random_state=self.random_state,
            n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=train_mean,
                name="Training Accuracy",
                mode="lines",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=test_mean,
                name="Validation Accuracy",
                mode="lines",
                line=dict(color="green"),
            )
        )
        fig.update_layout(
            title="Learning Curves",
            xaxis_title="Number of training examples",
            yaxis_title="Balenced Accuracy",
        )
        fig.show()

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
def get_higher_whisker(grp):
    try:
        q1, q3 = grp.quantile(0.25), grp.quantile(0.75)
        iqr = q3 - q1
        higher_whisker = q3 + (1.5*iqr)
        return higher_whisker
    except Exception as e:
        print("------------Error Occured-----------")
        print(traceback.format_exc())
        
def impute_higher_salary(df):
    
    df["higher_whisker"] = df.groupby('person_age')['person_income'].transform(lambda x: get_higher_whisker(x))
    df['higher_salary'] = df['person_income'] > df['higher_whisker']
    return df;

def define_home_owner(df):
    df['home_owner'] = df['person_home_ownership']=='OWN'
    return df

def define_loan_percentage(df):
    df['loan_percent_calc'] = (df['loan_amnt'] / df['person_income'])*100
    return df

def define_lower_loan_requirement(df):
    df['lower_loan_requirement'] =  df['loan_percent_calc'] < 20
    return df

def define_higher_loan_requirement(df):
    df['higher_loan_requirement'] =  df['loan_percent_calc'] > 40
    return df

def define_long_employement(df):
    df['long_working'] = df['person_emp_length'] > 10
    return df

def impute_empty_to_wrong_age(df):
        
    filter_con = df["person_age"] > 85
    df.loc[filter_con, 'person_age'] = np.nan
    return df

def impute_empty_to_wrong_employee_len(df):

    filter_con = df["person_emp_length"]  > 60
    df.loc[filter_con, 'person_emp_length'] = np.nan
    return df

def add_features(df):
    df = impute_empty_to_wrong_age(df)
    df = impute_empty_to_wrong_employee_len(df)
    df = impute_higher_salary(df)
    df = define_home_owner(df)
    df = define_loan_percentage(df)
    df = define_lower_loan_requirement(df)
    df = define_higher_loan_requirement(df)
    df = define_long_employement(df)
    return df

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
feature_engieered = ("FeatureEngineering_add_features", DataframeFunctionTransformer(add_features))

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# %%time
# Adaboost (Boosting of multiple Decision Trees)
from sklearn.ensemble import AdaBoostRegressor
ml_support_obj = ml_support()
best_decision_tree_model = RandomForestClassifier(random_state = 42,n_estimators=1200,
                                          min_samples_split=5, min_samples_leaf = 1,
                                          max_features = 'auto', max_depth = None,
                                          bootstrap =True
                                         )
regression_model = AdaBoostRegressor()
ct = ml_support_obj.get_column_transformer(False, "", True)
X_train, X_test , y_train, y_test = ml_support_obj.get_train_test_data()
finalized_pipeline = ml_support_obj.get_final_pipeline(regression_model,ct,feature_engieered)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
# %%time
preds = ml_support_obj.predit_with_pipeline(finalized_pipeline,X_train,X_test,y_train)

#%%
# --- [CELL 9]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
# === BEFORE (original) ===
# print("Balenced Accuracy Score : {0}".format(balanced_accuracy_score(y_test, preds)))

# === AFTER (edited) ===
binary_preds = np.where(preds > 0.5, 1, 0)
print("Balenced Accuracy Score : {0}".format(balanced_accuracy_score(y_test, binary_preds)))