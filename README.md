# Stock-Data-Analysis
Analyzed and predicted stock prices from two datasets using both supervised and unsupervised Machine Learning techniques using PySpark

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

## XGBOOST

### SCOPE

- XGBoost was used as a classification model
- Avoids **overfitting** by its regularization parameters
- Can deal with missing values and NaN values

### Methodology

- Predicted the closing price for the next 14 days on the validation data during training.
- Made predictions on holdout testing data.

### Observations

- We used **Root Mean Squared Error (RMSE)** function to check for prediction errors. Our calculations showed an average RMSE of 30 for all the stocks with the maximum reaching 77 for an individual stock.
- Validation prediction results sometimes varied by a big margin as it produced a RMSE very different from what we got in the testing data. For instance, TITAN, a stock in the dataset, had a difference of **47** in RMSE.




We found incomplete records for training and testing data.

**Training data -> 5.21%**
**Testing data -> 19.37%**

### Interpretation 

- For good predictions, RMSE is usually in the range 0.2-0.5.
- RMSE was very high on training compared to testing which indicates that the model did not learn the training distribution well.
- A more complex model could reduce underfitting

## CLUSTERING MODELS

- Time series k-means with Dynamic Time Warping
- Why use DTW instead of Euclidean?
  - Euclidean doesn’t work well if the time series lengths are mismatched
DTW is uses the nearest neighbour in the comparison sequence at each inde


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
