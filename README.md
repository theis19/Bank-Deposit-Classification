# Bank-Deposit-Classification

## Problem
Our Bank Marketing Team has trouble in getting more people to deposit their money. They have tried their best but of all the customers they reached out to, only around half of them deposits their money. As Benjamin Franklin would say **Time is money**, use data to redict if a customer will deposit their money in the bank and find out features of a customer that are more likely to deposit their money.

## Desc:
- **age**: age of the client
- **job**: type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
- **marital**: marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
- **education**: (categorical: primary, secondary, tertiary and unknown)
- **default**: has credit in default? (categorical: 'no','yes','unknown')
- **housing**: has housing loan? (categorical: 'no','yes','unknown')
- **loan**: has personal loan? (categorical: 'no','yes','unknown')
- **balance**: Balance of the individual.
- **contact**: contact communication type (categorical: 'cellular','telephone')
- **month**: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
- **day**: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
- **duration**: last contact duration, in seconds (numeric).
- **campaign**: number of contacts performed during this campaign and for this client (numeric, includes last contact)
- **pdays**: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
- **previous**: number of contacts performed before this campaign and for this client (numeric)
- **poutcome**: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
- **deposit** : has the client subscribed a term deposit? (binary: 'yes','no')

## Dataset Source
https://www.kaggle.com/janiobachmann/bank-marketing-dataset

## Exploratory Data Analysis (EDA)
- The marketing team targeted the campaign mostly for people with management and blue-collar job, but most of the people with blue-collar job didn't deposit in the end
- Retired people and students have higher chance of depositing their money, maybe we can shift some of our marketing target campaign to them.
- Single people are more likely to deposit their money than married people.
- People with no housing loan are more likely to deposit
- People with successful previous outcome from the previous campaign are more likely to deposit their money again, while people with failure from previous campaign outcome is more 50-50, this means they will likely try to deposit money even if they didn't want to before.
- The average age of clients is 41 years old, with older people more likely to deposit.

## Correlation Heatmap
By turning some categorical columns into ordinal data, we can use correlation to see the relationship of the features to the target (deposit). 
We could see that :
- The duration feature correlates the most with the target where the higher the duration the more likely the customer will deposit, but this seems a bit weird to me since it seems like if duration = 0 then deposit will 100% be **no** and this feature might leak the data to the machine learning algorithm.
- **pdays** correlates with the target where the longer the days they're previously contacted the more likely the customer will deposit. This doesn't really makes sense and below i will be removing this feature because there are too many data that has **-1** in it, which stands for never been contacted in previous campaign. The **-1** entries cause the data distribution to skew. Also it is highly correlated to **previous** feature which is why i will remove it.
- **previous** feature somewhat correlates with the target where customers that has been contacted in other previous campaigns are more likely to deposit.
- Customers that doesn't have any sort of loan are more likely to deposit, this is shown from the **loan** and **housing** features correlation towards the target.
- Not really significant but the more we contact the customer during a campaign the less likely the customer will deposit. Maybe we should try other tactic instead of being persistent towards one customer.
- Also not significant but the higher the customer's balance the more likely they will deposit. Which makes sense since they will have extra money to deposit anyway.

![alt text](https://github.com/theis19/Bank-Deposit-Classification/blob/master/images/corr.png "Correlation Heatmap")

## Model Building
Here i use 3 kind of models to train which are simple yet powerful for classification problem:
- Logistic Regression
- Random Forest
- Linear SVM

## Model Performance
**LogisticRegression** 
- Best Score :  0.8262964670750719
- Best Params :  {'clf__solver': 'saga', 'clf__penalty': 'none', 'clf__max_iter': 100}
- Accuracy Score on Test Data :  0.8240035826242723

**RandomForest** 
- Best Score :  0.8462305325265606
- Best Params :  {'clf__n_estimators': 100, 'clf__max_depth': 20}
- Accuracy Score on Test Data :  0.8454993282579489

**LinearSVC** 
- Best Score :  0.8236083324707263
- Best Params :  {'clf__penalty': 'l2', 'clf__max_iter': 1000}
- Accuracy Score on Test Data :  0.8204209583519928

## Model Evaluation
Model chosen for this is Random Forest, it achieved better accuracy than the other 2 algorithms.
The result is good with 84% accuracy, the model could predict both of the result whether it's a **yes** or **no** for deposit.

But it seems like the **duration** feature determines the most from the model's prediction. The reason this attribute highly affects the output target because the duration is not known before a call is performed (e.g., if duration=0 then y='no'). Also, after the end of the call y is obviously known. That's why we will remove this feature and run the model again.

![alt text](https://github.com/theis19/Bank-Deposit-Classification/blob/master/images/conf1.png "Confusion Matrix")
![alt text](https://github.com/theis19/Bank-Deposit-Classification/blob/master/images/feat_imp.png "Feature Importance")

## Improving the model by removing a feature
The accuracy score decreased but if we look closely on the confusion matrix, the True Positive doesn't change much, while the True Negative reduced by significant amount. Which means that the **duration** feature cause a data leak where **duration = 0 is always no**. This also means that the **duration** feature shouldn't really affect the machine learning prediction and therefore our machine learning algorithm will be better and know what features helps in determining the deposit prediction.

![alt text](https://github.com/theis19/Bank-Deposit-Classification/blob/master/images/conf2.png "Confusion Matrix")
![alt text](https://github.com/theis19/Bank-Deposit-Classification/blob/master/images/feat_imp2.png "Feature Importance")

This is more useful and we could tell our Marketing team to target :
- Customer with high balance
- Younger customers
- Customer with no loans, especially housing loan
- Do more campaigns in March
