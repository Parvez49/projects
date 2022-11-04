from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_protect

# Create your views here.

from django.template import loader
def index(request):
    temp={'Gender': None,
     'Married': None,
     'Dependents': None,
     'Education': None,
     'Self_Employed': None,
     'ApplicantIncome': None,
     'CoapplicantIncome': None,
     'LoanAmount': None,
     'Loan_Amount_Term': None,
     'Credit_History': None,
     'Property_Area': None
     }
    con={'a':'' , 'info': temp}
    #template = loader.get_template('loan.html')
    #return HttpResponse(template.render())
    return render(request,'loan.html',con)

#@csrf_protect
def predictLoan(request):
#def predictLoan():
    #temp = dict()
    if request.method=='POST':
    #if True:
        """
        temp['Gender']=int(request.POST.get('gender'))
        temp['Married'] = int(request.POST.get('married'))
        temp['Dependents'] = int(request.POST.get('dependents'))
        temp['Education'] = int(request.POST.get('education'))
        temp['Self_Employed'] = int(request.POST.get('self_employed'))
        temp['ApplicantIncome'] = float(request.POST.get('applicantIncome'))
        temp['CoapplicantIncome'] = float(request.POST.get('coappIn'))
        temp['LoanAmount'] = float(request.POST.get('loanamount'))
        temp['Loan_Amount_Term'] = float(request.POST.get('loanAmountTerm'))
        temp['Credit_History'] = int(request.POST.get('creditHistory'))
        temp['Property_Area'] = int(request.POST.get('propertyArea'))
        """
        temp={'Gender': int(request.POST.get('gender')),
        'Married': int(request.POST.get('married')),
        'Dependents': int(request.POST.get('dependents')),
        'Education': int(request.POST.get('education')),
        'Self_Employed': int(request.POST.get('self_employed')),
        'ApplicantIncome': float(request.POST.get('applicantIncome')),
        'CoapplicantIncome': float(request.POST.get('coappIn')),
        'LoanAmount': float(request.POST.get('loanamount')),
        'Loan_Amount_Term': float(request.POST.get('loanAmountTerm')),
        'Credit_History': int(request.POST.get('creditHistory')),
        'Property_Area': int(request.POST.get('propertyArea'))
              }

        print(temp)

        import numpy as np
        import math
        import numpy

        import pandas as pd
        from sklearn.model_selection import train_test_split
        import sklearn
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import RandomForestRegressor


        train = pd.read_csv("Loan/train.csv")
        test = pd.read_csv("Loan/test.csv")
        ss = pd.read_csv("Loan/sample.csv")

        data = pd.concat([train, test])

        data = pd.concat([train, test])
        # print(data)
        data.drop("Loan_ID", axis=1, inplace=True)
        for i in [data]:
            i["Gender"] = i["Gender"].fillna(data.Gender.dropna().mode()[0])
            i["Married"] = i["Married"].fillna(data.Married.dropna().mode()[0])
            i["Dependents"] = i["Dependents"].fillna(data.Dependents.dropna().mode()[0])
            i["Self_Employed"] = i["Self_Employed"].fillna(data.Self_Employed.dropna().mode()[0])
            i["Credit_History"] = i["Credit_History"].fillna(data.Credit_History.dropna().mode()[0])
            i["Loan_Amount_Term"] = i["Loan_Amount_Term"].fillna(data.Loan_Amount_Term.dropna().mode()[0])
            i["LoanAmount"] = i["LoanAmount"].fillna(data.LoanAmount.dropna().mode()[0])
            i["Loan_Status"] = i["Loan_Status"].fillna(data.Loan_Status.dropna().mode()[0])

            # i["EMI_per_Loan_Amount_Term"]=i["EMI_per_Loan_Amount_Term"].fillna(data.EMI_per_Loan_Amount_Term.dropna().mode()[0])
            # i["EMI_per_LoanAmount"]=i["EMI_per_LoanAmount"].fillna(data.EMI_per_LoanAmount.dropna().mode()[0])
        # print(data)
        #data.isnull().sum()

        data1 = data.loc[:, ['LoanAmount', 'Loan_Amount_Term']]
        imp = IterativeImputer(RandomForestRegressor(), max_iter=10, random_state=0)
        data1 = pd.DataFrame(imp.fit_transform(data1), columns=data1.columns)
        # print(data1)

        for i in [data]:
            i["Gender"] = i["Gender"].map({"Male": 0, "Female": 1}).astype(int)
            i["Married"] = i["Married"].map({'No': 0, "Yes": 1}).astype(int)
            i["Education"] = i["Education"].map({"Not Graduate": 0, "Graduate": 1}).astype(int)
            i["Self_Employed"] = i["Self_Employed"].map({'No': 0, "Yes": 1}).astype(int)
            i["Credit_History"] = i["Credit_History"].astype(int)

        for i in [data]:
            i["Property_Area"] = i["Property_Area"].map({"Urban": 0, "Rural": 1, "Semiurban": 2}).astype(int)
            i["Dependents"] = i["Dependents"].map({"0": 0, "1": 1, "2": 2, "3+": 3}).astype(int)

        # new_train=data.iloc[:614]
        # new_test=data.iloc[614:]
        new_train = data.iloc[:800]
        new_test = data.iloc[800:]

        new_train["Loan_Status"] = new_train["Loan_Status"].map({'N': 0, "Y": 1}).astype(int)

        for i in [data]:
            i["TotalIncome"] = i["ApplicantIncome"] + i["CoapplicantIncome"]

        r = 0.00833
        data['EMI'] = data.apply(lambda x: (x['LoanAmount'] * r * ((1 + r) ** x[('Loan_Amount_Term')])) / (
                    (1 + r) ** ((x['Loan_Amount_Term']) - 1)), axis=1)

        data['Dependents_EMI_mean'] = data.groupby(['Dependents'])['EMI'].transform('mean')
        data['LoanAmount_per_TotalIncome'] = data['LoanAmount'] / data['TotalIncome']
        data['LoanAmount_per_TotalIncome'] = data['Loan_Amount_Term'] / data['TotalIncome']
        data['EMI_per_Loan_Amount_Term'] = data['EMI'] / data['Loan_Amount_Term']
        data['EMI_per_LoanAmount'] = data['EMI'] / data['LoanAmount']
        data['Property_Area_LoanAmount_per_TotalIncome_mean'] = data.groupby(['Property_Area'])[
            'LoanAmount_per_TotalIncome'].transform('mean')
        data['Credit_History_Income_Sum'] = data.groupby(['Credit_History'])['TotalIncome'].transform('sum')
        data['Dependents_LoanAmount_Sum'] = data.groupby(['Dependents'])['LoanAmount'].transform('sum')

        from sklearn.preprocessing import KBinsDiscretizer
        Loan_Amount_Term_discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        data['Loan_Amount_Term_Bins'] = Loan_Amount_Term_discretizer.fit_transform(
            data['Loan_Amount_Term'].values.reshape(-1, 1)).astype(float)

        TotalIncome_discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        data['TotalIncome_Bins'] = TotalIncome_discretizer.fit_transform(
            data['TotalIncome'].values.reshape(-1, 1)).astype(float)
        LoanAmount_per_TotalIncome_discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        data['LoanAmount_per_TotalIncome_Bins'] = LoanAmount_per_TotalIncome_discretizer.fit_transform(
            data['LoanAmount_per_TotalIncome'].values.reshape(-1, 1)).astype(float)

        data = data.drop(['EMI'], axis=1)
        data = data.drop(['TotalIncome'], axis=1)
        data = data.drop(['LoanAmount_per_TotalIncome'], axis=1)

        #new_train.shape

        x = new_train.drop("Loan_Status", axis=1)
        y = new_train["Loan_Status"]
        #new_train.head()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        #print(x_train.shape)
        #print(x_test.shape)

        from sklearn import linear_model
        from sklearn.metrics import accuracy_score
        log_clf = linear_model.LogisticRegression()
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer

        """
        dd = pd.DataFrame({
            'Gender': 0,
            'Married': 0, 'Dependents': 3, 'Education': 1, 'Self_Employed': 0, 'ApplicantIncome': 9167,
            'CoapplicantIncome': 0,
            'LoanAmount': 185, 'Loan_Amount_Term': 360, 'Credit_History': 1, 'Property_Area': 1
        }, index=[0])
        """

        dd=pd.DataFrame.from_dict([temp])

        predo = log_clf.fit(x_train, y_train).predict(x_test)
        # print(predo)
        # print(y_test[:50])
        accuracy_score(predo, y_test)
        print(x_train.shape)
        print(log_clf.score(x_train, y_train))

        cross_val_score(log_clf, x_train, y_train, scoring=make_scorer(accuracy_score), cv=3)

        #array([0.8041958, 0.7902097, 0.7972028])
        predo = log_clf.fit(x_train, y_train).predict(dd)
        print(x_test.head(1))
        print(predo)
        #acr = accuracy_score(predo, y_test)

        #print("Accuracy= ", acr * 100, "%")




        con = {'a': int(predo[0]),'info':temp}
        #con2={'b':5}
        return render(request,'loan.html',con)
    #print(temp)
    template = loader.get_template('loan.html')
    return HttpResponse(template.render())
#predictLoan()
def mlcode():
    pass