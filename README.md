# Online-payment-fraud-detection
Online payment fraud detection plays a vital role in enhancing security, building trust, and ensuring the reliability of online payment systems for both businesses and consumers.
Online payment fraud detection is critical for safeguarding customers, businesses, and the integrity of digital payment systems. By detecting and preventing fraudulent transactions, businesses can protect customers from financial losses, identity theft, and unauthorized access to their accounts. 
Implementing advanced fraud detection techniques, such as machine learning algorithms and behavioral analytics, enables businesses to stay ahead of evolving fraud tactics and adapt to new threats.

# DEMO VIDEO:

## Objective
To identify fraudulent and non-fraudulent payments.
## About dataset
To identify online payment fraud with machine learning, we need to train a machine learning model for classifying fraudulent and non-fraudulent payments. For this, we need a dataset containing information about online payment fraud, so that we can understand what type of transactions lead to fraud. For this task, I collected a dataset from Kaggle, which contains historical information about fraudulent transactions which can be used to detect fraud in online payments. Below are all the columns from the dataset I’m using here:
The dataset consists of 10 variables:
- step: represents a unit of time where 1 step equals 1 hour
- type: type of online transaction
- amount: the amount of the transaction
- nameOrig: customer starting the transaction
- oldbalanceOrg: balance before the transaction
- newbalanceOrig: balance after the transaction
- nameDest: recipient of the transaction
- oldbalanceDest: initial balance of recipient before the transaction
- newbalanceDest: the new balance of recipient after the transaction
- isFraud: fraud transaction
## Framework
Implementing an online payment fraud detection project is important for a few key reasons:
- Fraudsters are always trying new tricks. By keeping up with the latest methods, you can stay one step ahead and protect yourself and your customers.
- Models like logistic regression and svc are used for gaining knowledge or patterns from the information contained within a dataset.
### Data reading
The dataset has been downloaded from kaggle.The first step is reading data using pandas (pd.read_csv("")).
## Data preprocessing
- performed encoding for categorical variable of type column
- Analyzed the data of single variable.
- Using Quantile-based Flooring and Capping removed outliers for those columns
- Capping is replacing all higher side values exceeding a certain theoretical maximum or upper control limit (UCL) by the UCL value. Here we'll do 90th percentile for higher values.Flooring is replacing all values falling below a certain theoretical minimum or lower control limit (UCL) by the LCL value. Here we'll do 90th percentile for higher values.
- Since the dataset is too large, standardscaler is used.
### Models used:
I used logistic regression,svc,XGBoost for predicting fraudulent transaction.
### Evaluation metrics
This model was evaluated with metrics accuracy().
## Result
- The best performance was achieved using svc with accuracy rate of around 99%.
