# Airline_passenger_satisfaction

The data set you will use is the airline customer satisfication data set found on Kaggle
[here](https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction)

The original data set consists of 103904 observations, and my goal is to predict the satisfaction. I hold out part of the observations and only used 40000 of the observation (air training.csv) to train the
models. Performed prediction on a test set. 

## More Details:

Used classifiers SVM, Neural Network, and Naive Bayes along with ensemble methods. Built classifier models using each of these models (hence 4 classifiers total) using the provided data set (again, to predict the satisfaction).

For ensemble, built a stacking model. Performed feature engineering, feature
selection, etc)

## Model contains following:
1. cleaned up the features and check for any missing values
2. tuning procedure for each method
3. evaluation of each method (based on a hold out set).
4. The four models I used to build are SVM, Neural Network, Naive Bayes, and Stacking.
