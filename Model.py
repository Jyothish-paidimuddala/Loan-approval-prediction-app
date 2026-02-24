import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

data = pd.read_csv("dataset.csv")

le = LabelEncoder()

data['Gender'] = le.fit_transform(data['Gender'])
data['Married'] = le.fit_transform(data['Married'])
data['Education'] = le.fit_transform(data['Education'])
data['Loan_Status'] = le.fit_transform(data['Loan_Status'])

X = data[['Gender','Married','Education','ApplicantIncome','LoanAmount','Credit_History']]
y = data['Loan_Status']

model = LogisticRegression()
model.fit(X,y)

pickle.dump(model, open('model.pkl','wb'))
