# Stock-Data-Analysis
Analyzed Stock data set using both supervised and unsupervised Machine learning techniques using PySpark

## Table of Contents  
- [XGBoost](#xgboost)      
<a name="xgboost"/>


- [Clustering Models](#clustering-models)      
<a name="clustering-models"/>

- [Summary](#summary)      
<a name="summary"/>

- [Possible techniques that can improve the accuracy](#possible-techniques-that-can-improve-the-accuracy)      
<a name="possible-techniques-that-can-improve-the-accuracy"/>


---------------------------------------------------------------------------------------------------------------


The goal was to predict the class of ~380 observations (proteins) in the test file. The data were messy, and had missing values. It was also not obvious how to include information from the additional file Protein_interactions.csv, into the training data.

## DATA PREPROCESSING

### INCOMPLETE RECORDS

<img width="390" alt="Screen Shot 2021-04-25 at 5 16 07 PM" src="https://user-images.githubusercontent.com/43936803/115996967-0d5c1180-a5ea-11eb-874a-b73ac09ef41c.png">

We found incomplete records for training and testing data.

**Training data -> 5.21%**
**Testing data -> 19.37%**

Incomplete variables were also present.

- Chromosome
- Essential

### FEATURE ENGINEERING

- Sum of binary variables
  - Summarize the 442 binary variables by sum where True -> 1

- Dummy Variables  (Performance: **High**)
  -Converted Chromosome, Essential, and Interaction Type into Binary Variable columns

- Average Interaction Strength (Performance: **High**)
  - Average of all interaction strengths for each protein

- Dominant Interaction Type (Performance: **Low**)
  - Most popular interaction type for each protein

## MODEL PROPOSAL
- Random Forest
- Support Vector Machine
- XGBoost

### RANDOM FOREST

- Random Forest tends **not to overfit the data**
- Tuning parameters is simple and fast.
- Convenient for feature selection
- Moderate validation accuracy: **64% overall**
- Difficult to interpret the model and re-engineer the model/data

<img width="635" alt="Screen Shot 2021-04-25 at 5 06 59 PM" src="https://user-images.githubusercontent.com/43936803/115996698-0bde1980-a5e9-11eb-8d63-d1148cc00334.png">

### SUPPORT VECTOR MACHINE

#### GAUSSIAN KERNEL
- High validation accuracy: **88% overall**
- **Overfits the training data**: the model explained random noises in the training data
- Poor prediction result on Kaggle

#### LINEAR KERNEL
- Moderate validation accuracy: **83% overall**
- Moderately fast tuning by grid searching
- Low accuracy in less dominant labels


### XGBOOST

- Xgboost is a more efficient implementation of Gradient Boosting
- Xgboost multiclass model: **soft-max**
- Evaluation metric: **mlogloss**
- Encode categorical variables to multiple binary variables
- High validation accuracy: **84%**



## SUMMARY 

- Key Variables: 
  - chromosome_interac (our engineered variable)
  - Intracellular transport (binary variable)
- The model pathway : **RandomForest -> SVM Gaussian -> SVM linear -> XGBoost**
- Our best performing models were XGBoost and SVM Linear kernel. 
- From XGBoost, we learned that regularization is key to combat overfitting
- **Final Result: 61% accuracy**
- Model could be **underfit for the data**

## POSSIBLE TECHNIQUES THAT CAN IMPROVE THE ACCURACY
- **Hyperparameter Tuning** 
- **Stacking models**
