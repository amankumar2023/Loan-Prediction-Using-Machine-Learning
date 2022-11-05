from copyreg import pickle
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as mtp
import pickle
# print("0")
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#print("1")
#import warnings 
#if not sys.warnoptions:
 #   warnings.simplefilter("ignore")
#warnings.filterwarnings("ignore",category=DeprecationWarning)

dataset= pd.read_csv('../train_u6lujuX_CVtuZ9i (1).csv')

dataset_1=dataset.dropna()
dataset_2=dataset_1.drop(['Loan_ID'],axis=1)
dataset_3=pd.get_dummies(dataset_2)

print("2")
# Drop columns
dataset_4 = dataset_3.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 
              'Self_Employed_No', 'Loan_Status_N'], axis = 1)

# Rename columns name
dataset_4.rename(columns={'Gender_Male': 'Gender', 'Married_Yes': 'Married', 
       'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed',
       'Loan_Status_Y': 'Loan_Status'}, inplace=True)

x = dataset_4.drop(["Loan_Status"], axis=1)
y = dataset_4["Loan_Status"]

from imblearn.over_sampling import SMOTE
X, Y = SMOTE().fit_resample(x, y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
print("3")
lr=LogisticRegression(max_iter=500)
lr_1= lr.fit(X_train,Y_train)

pickle.dump(lr_1,open('model.pkl','wb'))

