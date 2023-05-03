<p align="center">
<img src="/Images/supervised-learning.jpg" alt="Supervised Learning">
</p>

# Module 19 Challenge Submission - Supervised Machine Learning

#### This repository contains the code and resources for the Credit Risk Evaluator project, developed to explore the use of machine learning models in evaluating credit risk for loan applicants. In this project, we have utilized the Pandas library for data manipulation and scikit-learn for building and comparing two popular classification models: Logistic Regression and Random Forests Classifier. We have applied these models to a lending dataset to predict the loan status of loan applicants. Through this project, we aim to provide a better understanding of how machine learning models can be used in credit risk evaluation and to compare the performance of different models.


## Table of Contents
- [Getting Started](#getting-started)
- [Project Deliverables](#project-deliverables)
- [Conclusion](#conclusion)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [References](#references)

## Getting Started

To get started with the Credit Risk Evaluator project, you will need to have Python 3 and several libraries Pandas and scikit-learn. You can install these libraries using pip or conda. Once you have the necessary libraries, you can retrieve the lending dataset, which is located in the Resources folder. The main code for the project is in the main.ipynb Jupyter Notebook, which contains the steps for data preparation, model creation, and model comparison. You can run the code cells in the notebook to replicate the analysis and generate the results.

## Project Deliverables

In this project, we will be creating two machine learning models to evaluate credit risk for loan applicants: a Logistic Regression model and a Random Forest Classifier model. We will compare the performance of these models to determine which one is better suited for this task.

To accomplish this, we will first split the data into training and testing sets using train_test_split from sklearn. We will then create, fit, and score both models using the training and testing sets.

```
# Split the data into X_train, X_test, y_train, y_test
X = data.drop("loan_status", axis=1)
y = data["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

# Train a Logistic Regression model and print the model score
lr_model = LogisticRegression(max_iter=100)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_score = accuracy_score(y_test, lr_pred)
print("Logistic Regression accuracy:", lr_score)

# Train a Random Forest Classifier model and print the model score
rf_model = RandomForestClassifier(n_estimators=100, random_state=18)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_score = accuracy_score(y_test, rf_pred)
print("Random Forest Classifier accuracy:", rf_score)

print("Logistic Regression accuracy:", lr_score)
print("Random Forest Classifier accuracy:", rf_score)
```

## Conclusion
Based on the results of our models, both the Logistic Regression and Random Forest Classifier models achieved similar results for this dataset. However, the Logistic Regression model was quicker as predicted, so it is better suited for this task. Therefore, we recommend using the Logistic Regression model for credit risk evaluation of loan applicants.

## Project Structure

```
README.md
index.html
Credit Risk Evaluator.ipynb
requirements.txt
images
   |-- supervised-learning.jpg
Resources
   |-- lending_data.csv

```
## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments
I would like to thank our bootcamp instructors for their guidance and support throughout this assignment.

## References
-	Pandas: https://pandas.pydata.org/
-	scikit-learn: https://scikit-learn.org/stable/
-	Jupyter Notebook: https://jupyter.org/
-	LendingClub dataset source: https://www.kaggle.com/wordsforthewise/lending-club
-	University of Western Australia Data Bootcamp: https://bootcamp.ce.uwa.edu.au/data/
