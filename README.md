# <p align = center> Machine Learning the Identities of Labour’s Electorate: Random Forest (Bagged Decision Trees) and XGBoost (Boosted Decision Trees) on BES Wave 30 (1,500 words) </p>

This project is intended to fulfil the requirements of Problem Set 1 of the [Machine Learning for Social Data Science](https://www.exeter.ac.uk/study/studyinformation/modules/info/?moduleCode=SSIM916&ay=2025/6&sys=0) module at the University of Exeter. 

It uses ensemble learning techniques to predict who voted for Labour in the 2024 General Election. The data comes from Wave 28 of the British Election Study (BES), a large national survey of UK citizens. 

&nbsp;

## :dart: Project Aim
The project answers two main questions:
1. Which social identity factors predict voting Labour?
2. Does Random Forest (bagging) or XGBoost (boosting) perform better? 

## :bar_chart: Dataset
**Source:** British Election Study (BES)  
**Wave:** 28 (2024 General Election Campaign)  
**Sample Size:** 30,342 respondents  
**Unit of Analysis:** Individual respondents  

### Target Variable
`Labour_voter`, a binary indicator (1 = voted Labour, 0 = did not vote Labour) derived from `General_Election_Vote`

### Features 
* Age
* Gender
* Ethnicity
* Religion
* Sexuality
* Disability Status
* Education level
* Subjective class
* Region

&nbsp;

## :lock: Ethics
* Uses anonymised secondary survey data
* No personal data is identifiable

&nbsp;

## :robot: Models Used
1. XGBoost (Boosted Decision Tree)  
Boosting builds sequentially. Each new tree corrects the mistakes of the previous ones.  

     * Model 1 (Baseline)  
     * Model 2 (Stratified train/test split)  
     * Model 3 (Class Weights)  
     * Model 4 (Adjusted Probability Threshold)   
     * Model 5 (Bayesian hyperparameter optimisation)  

2. Random Forest (Bagged Decision Tree)  
Bagging builds many independent trees on boostrapped samples and average predictions.  

     * Model 1 (Baseline)   
     * Model 2 (Stratified train/test split)  
     * Model 3 (Class Weights)   
     * Model 4 (Adjusted Probability Threshold)  
     * Model 5 (Hyperparameter tuning using `RandomizedSearchCV`)

*Model improvements applied step by step*  

&nbsp;

## :chart_with_upwards_trend: Evaluation Metrics
* Precision
* Recall
* F1
* PR-AUC

*Five-fold cross-validation was also performed to ensure robustness*

&nbsp;

## :balance_scale: Class Imbalance 
Only **17.7%** of respondents voted Labour.

This created a class imbalance problem. 

To address this, I used:
* Stratified train/test splits
* Applied class weights
* Adjusted the probability threshold
* PR-AUC was used instead of ROC-AUC

&nbsp;

## :mag_right: Key Findings
Across both models, the strongest predictors were:
* Age
* Education
* Region
* Religion
* Subjective class

This suggests Labour's 2024 coalition appears structured more around education and cultural cleavages than traditional class alone. 

**Model Comparison**
* Random Forest showed slightly more consistent performance under cross-validation
* Both models struggled with precision due to class imbalance
* Random Forest severely overpredicted Labour in the final confusion matrix
* XGBoost showed better balance but weaker specificity 

&nbsp;

## :warning: Limitations
* Findings are limited to the 2024 pre-election period
* Models show modest predictive power
* Severe class imbalance affects precision
* No formal statistical comparison test (e.g., McNemar's test) was conducted

&nbsp;

## :rocket: Future Improvements
* Model intersectional identity interactions explicitly
* Apply McNemar's test for statistical comparison

&nbsp;

## :hammer_and_wrench: Requirements
To run this project, install:
* Pandas
* Numpy
* Scikit-learn
* Xgboost
* Matplotlib
* Seaborn
