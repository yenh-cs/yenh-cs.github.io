# Data Science in Plain Text

Welcome to 'Data Science in Plain Text', a page dedicated to making data science accessible to everyone. 

Data science has become an increasingly important field, with applications ranging from business intelligence to healthcare and beyond. However, with all of the technical jargon and complex algorithms, it can often seem intimidating and inaccessible to those without a background in programming or statistics. 

That's where this blog comes in. My goal is to break down the concepts and tools of data science into plain language, making it easier for anyone to understand and apply to their work or personal projects. Whether you're a beginner just starting out or a seasoned data professional looking to expand your knowledge, my articles will cover a range of topics, from data cleaning and visualization to machine learning and beyond. I believe that everyone should have access to the power of data science, and I hope that this blog will serve as a valuable resource for anyone looking to learn more about this exciting field. 

Thank you for joining me on this journey to demystify data science and make it more accessible to everyone.


# Mathematics for ML
## Statistics 101
#### Measures of Central Tendency and Data Dispersion
The first stage of any machine learning project is to explore the data, and a crucial part of that exploration is understanding the measures of central tendency and dispersion. 

It's always a good idea to summarize the data by means of graphs:
* For *nominal/ordinal* data, i.e. colors of the rainbow, types of animals, and political affiliations or grades in school (A, B, C, D, F), customer satisfaction ratings (very satisfied, satisfied, neutral, dissatisfied, very dissatisfied), and clothing sizes (small, medium, large) -> bar graph.
* For *interval ratio varibale* (numeric data such house price) -> histogram.

Assessing the shape of a distribution is of essential importance because it could effect the statistical methods that we can employ. 

Besides summarizing data by charts, it's useful to describe the center of a distribution. **Measures of central tendency**, such as the *mean*, *median*, and *mode*, provide insight into the "center" of a dataset in which:
* **Mode**: The mode is the most frequently occurring value in a dataset. It can be calculated by simply counting the number of times each value appears in the dataset and finding the value with the highest count.

* **Mean**: The mean is the average value of a dataset. It can be calculated by summing up all the values in the dataset and dividing by the total number of values.
```
Mean = (sum of all values) / (total number of values)
```
* **Median**: The median is the middle value in a dataset when the values are arranged in ascending or descending order. It can be calculated by finding the value that separates the dataset into two equal halves. For an odd number of values in the dataset, the median is the middle value. For an even number of values, the median is the average of the two middle values. 
Understanding these measures is important for machine learning applications because they help us identify trends and patterns in the data. For example, if we are analyzing customer behavior data, we might use the mean to understand the average amount of time customers spend on our website. We might use the standard deviation to determine how much the time spent varies from the mean and to identify any outliers in the data. 

To adequately describe a distribution, we also need to explore the variability or dispersion of data via **Measures of dispersion**, including *range*, *interquartile range*, *variance*, and *standard deviation* in which
* **Range**: The range is the difference between the highest and the lowest value in a dataset. It provides a measure of how spread out the data is.
```
Range = max(x) - min(x)
```

* **Interquartile Range (IQR)**: The IQR is the range of the middle 50% of the dataset, which is the difference between the third quartile (75th percentile) and the first quartile (25th percentile). It is less affected by extreme values than the range and provides a measure of the spread of the central part of the data.
```
Interquartile Range (IQR) = Q3 - Q1, where Q1 is the first quartile (25th percentile) and Q3 is the third quartile (75th percentile)
```

* **Variance**: The variance is a measure of how much the data deviates from its mean. It is calculated by finding the average of the squared differences between each value and the mean. A higher variance indicates that the data is more spread out.
```
Variance = (1/n) * Σ(xi - x̄)^2, where n is the number of observations, Σ is the sum of, xi is the i-th observation, and x̄ is the sample mean.
```

* **Standard Deviation**: The standard deviation is the square root of the variance and is commonly used to measure the spread of data in a normal distribution. It provides a measure of how much the data deviates from its mean in the same unit of measurement as the data. A higher standard deviation indicates that the data is more spread out.
```
Standard Deviation = sqrt(1/n * Σ(xi - x̄)^2), where sqrt denotes the square root and all the other symbols are the same as in the variance formula.
```

One useful tool in a data scientist's tool box is **Box plots** which are an essential tool for visualizing the measures of central tendency and dispersion. They offer a simple way to compare multiple datasets and identify differences between them, making it easy to spot outliers. By analyzing the quartiles of data based on the median, box plots can provide insights into the data's characteristics and identify potential problems. For example, suppose we are analyzing the performance of two different machine learning models. We can use a box plot to compare the accuracy scores of the two models. If the box plot shows that the median score of the first model is higher than the second model, we can conclude that the first model performs better. If the box plot shows that the second model has a wider range, we can investigate further to identify any outliers or potential issues with the data.

![This is a box plot.](/assets/images/boxplot.png "This is a box plot.")

It should pay attention that the the area below the median is greater than the area above the median line in a box plot. It means that the lower half of the data set is more spread out than the upper half of the data set. This can happen when the distribution of the data is skewed, with a longer tail on one side of the median than the other. The box plot shows the quartiles of the data, which are based on the median, so if the data is skewed, the lower quartile (Q1) and upper quartile (Q3) may be closer together on one side of the median than the other. This can result in a longer box on one side of the median and a shorter box on the other side, with more outliers on the longer side. In general, if the area below the median is much greater than the area above the median line in a box plot, it suggests that there is more variability in the lower half of the data set than in the upper half. However, the exact interpretation depends on the shape of the distribution and the nature of the data being analyzed.


## Linear Algebra
Placeholder

## Multivariate Calculus
Placeholder


# Machine Learning
## Supervised Learning
#### A Comprehensive Template For Classification Problem
1. **Data Preprocessing**
* Remove any duplicate records.
* Handle missing values using techniques such as imputation or deletion. 
* Handle any outliers by either removing them or replacing them with more appropriate values. 
* Check for any data inconsistencies and correct them. 
* Convert categorical variables into numerical values (If needed). 
* Normalize or standardize the numerical variables (if needed). 
* Perform feature scaling to avoid any bias towards features with higher magnitudes (if needed).

2. **Exploratory Data Analysis (EDA)**
* Visualize the data to identify any patterns or trends. 
* Explore the distribution of the variables.
* Look for any correlations between variables Identify any outliers or anomalies. 
* Consider exploring the data in more depth by examining the distribution of each variable by class, identifying any nonlinear relationships, and checking for any seasonal or time-series patterns.

3. **Feature Selection**
* Identify the relevant features that contribute most to the outcome variable. 
* Remove any irrelevant or redundant features that do not contribute significantly. 
* Use techniques such as feature importance, correlation matrix, or PCA.
* Consider using wrapper methods, such as recursive feature elimination, which can perform a more exhaustive search for relevant features. 

4. **Feature Engineering**
* Create new features that may be more informative or relevant. 
* Transform existing features to make them more suitable for modelling. 
* Use techniques such as one-hot encoding, binning, or scaling.
* Consider using domain expertise to create features that may be more relevant and meaningful. For example, in the medical domain, it may be useful to create features that capture patient demographics, medical history, and lifestyle factors. 

5. **Data Standardization**
* Scale the numerical features to have a mean of 0 and a standard deviation of 1. 
* Helps the model to converge faster and improves the model's performance.
* Consider using more advanced techniques such as min-max scaling or robust scaling to handle outliers or non-Gaussian distributions. 

6. **Data Split**
* Split the data into training, validation, and test sets 
* Use a stratified sampling technique to ensure that each class is represented equally in each set 

7. **Process Imbalanced Data Set**
* Identify any class imbalance in the dataset. 
* Use techniques such as oversampling or undersampling to balance the classes. 
* Use techniques such as SMOTE or ADASYN to generate synthetic samples for the minority class. 
* Consider using ensemble methods such as bagging or boosting to handle class imbalance, as they can improve the performance of the model on the minority class. 

8. **Model Training**
* Train the model using a suitable algorithm (e.g., logistic regression, decision trees, random forests, SVM, or neural networks). 
* Use the training set to train the model and tune the hyperparameters.
* Consider using ensemble methods, such as random forests or gradient boosting, which can improve the model's performance by combining multiple weak learners. 

9. **Model Validation**
* Use the validation set to evaluate the performance of the model. 
* Use metrics such as accuracy, precision, recall, F1 score, ROC curve, or AUC.
* Consider using cross-validation, which can provide a more robust estimate of the model's performance by using multiple validation sets. 

10. **Model Tuning**
* After selecting the model, the next step is to fine-tune its hyperparameters. Hyperparameters are model settings that cannot be learned during training, and their values must be set by the data scientist. Hyperparameters can have a significant impact on the model's performance, and tuning them correctly can improve the model's accuracy. Techniques such as grid search or random search can be used to find the best combination of hyperparameters that maximize the performance of the model.
* Consider using Bayesian optimization, which can perform a more efficient search for optimal hyperparameters compared to grid search or random search. 

11. **Model Testing**
* Once the model is trained and tuned, it is evaluated on the test set to measure its final performance. The test set should be kept completely separate from the training and validation sets to avoid overfitting. 
* The model's performance on the test set provides an unbiased estimate of how well the model will perform on new data.

12. **Interpretation**
* After building a model, it is important to understand how it is making predictions. This can be done by analyzing the model's feature importance, decision boundaries, and activation maps. Feature importance can identify which features are contributing the most to the model's predictions, while decision boundaries can reveal how the model is separating the different classes. Activation maps can help visualize how the model is processing and transforming the input data.
* Consider using more advanced techniques such as SHAP values or LIME to understand how the model is making predictions.

13. **Deployment**
* Once the model is built and tested, it can be deployed in a production environment to make real-world predictions. The deployment process involves integrating the model into a larger system and making sure it is accessible and easy to use by other stakeholders. Depending on the specific use case, the model may need to be integrated into a web application, mobile app, or other software system. 
* Consider using containerization technologies such as Docker to make deployment easier and more efficient.

14. **Monitoring**
* After the model is deployed, it is important to monitor its performance to ensure it continues to perform well. Monitoring involves tracking the model's performance metrics and making adjustments as needed. If the model's performance starts to degrade, it may be necessary to retrain the model on new data or update its hyperparameters. 
* Consider setting up alerts or triggers to notify stakeholders if the model's performance starts to degrade.

15. **Maintenance**
* As new data becomes available or as the environment changes, it may be necessary to update the model to ensure it continues to provide accurate predictions. Maintenance involves updating the model's training data, retraining the model, and updating its hyperparameters as needed. It is important to regularly evaluate the model's performance and make adjustments to ensure it remains accurate and relevant over time.
* Consider using automated pipelines to handle data updates and model retraining, which can save time and improve efficiency.


#### A Comprehensive Template For Regression Problem
1. **Data Preprocessing**
* Remove any duplicate records.
* Handle missing values using techniques such as imputation or deletion.
* Handle any outliers by either removing them or replacing them with more appropriate values. 
* Check for any data inconsistencies and correct them. 
* Convert categorical variables into numerical values (if needed).

2. **Exploratory Data Analysis (EDA)**
* Visualize the data to identify any patterns or trends. 
* Explore the distribution of the variables. 
* Look for any correlations between variables 
* Identify any outliers or anomalies. 
* Consider exploring the data in more depth by examining the distribution of each variable, identifying any nonlinear relationships, and checking for any seasonal or time-series patterns.

3. **Feature Selection**
* Identify the relevant features that contribute most to the outcome variable. 
* Remove any irrelevant or redundant features that do not contribute significantly.
* Use techniques such as feature importance, correlation matrix, or PCA.
* Consider using wrapper methods, such as recursive feature elimination, which can perform a more exhaustive search for relevant features.

4. **Feature Engineering**
* Create new features that may be more informative or relevant.
* Transform existing features to make them more suitable for modelling.
* Use techniques such as one-hot encoding, binning, or scaling.
* Consider using domain expertise to create features that may be more relevant and meaningful.

5. **Data Standardization**
* Scale the numerical features to have a mean of 0 and a standard deviation of 1.
* Helps the model to converge faster and improves the model's performance.
* Consider using more advanced techniques such as min-max scaling or robust scaling to handle outliers or non-Gaussian distributions.

6. **Data Split**
* Split the data into training, validation, and test sets.
* Use a stratified sampling technique to ensure that each class is represented equally in each set.

7. **Model Selection**
* Choose a suitable regression algorithm (e.g., linear regression, polynomial regression, support vector regression, decision trees, random forests, or neural networks).
* Consider using ensemble methods, such as random forests or gradient boosting, which can improve the model's performance by combining multiple weak learners.

8. **Model Training**
* Train the model using the training set.
* Tune the hyperparameters of the model using techniques such as grid search or random search.

9. **Model Validation**
* Use the validation set to evaluate the performance of the model.
* Use metrics such as mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), or R-squared (R2).
* Consider using cross-validation, which can provide a more robust estimate of the model's performance by using multiple validation sets.

10. **Model Tuning**
* After selecting the model, the next step is to fine-tune its hyperparameters. Hyperparameters are model settings that cannot be learned during training, and their values must be set by the data scientist. Hyperparameters can have a significant impact on the model's performance, and tuning them correctly can improve the model's accuracy. Techniques such as grid search or random search can be used to find the best combination of hyperparameters that maximize the performance of the model. * Consider using Bayesian optimization, which can perform a more efficient search for optimal hyperparameters compared to grid search or random search.

11. **Model Testing**
* Once the model is trained and tuned, it is evaluated on the test set to measure its final performance. The test set should be kept completely separate from the training and validation sets to avoid overfitting. The model's performance on the test set provides an unbiased estimate of how well the model will perform on new data.

12. **Interpretation**
* Once the model is trained, and its performance has been evaluated, it's time to interpret the results. 
* Identify which features have the most significant impact on the outcome variable. 
* Use techniques such as feature importance, partial dependence plots, or permutation feature importance to interpret the model's behaviour. 
* Consider using SHAP (SHapley Additive exPlanations), which provides a unified framework for interpreting the output of any machine learning model. SHAP values explain the output of any model by computing the contribution of each feature to the prediction. 
* Visualize the results to communicate the findings effectively. Present the results in a way that is easily understandable to stakeholders and decision-makers.

13. **Deployment** 
* Once the model is developed and tested, the next step is to deploy it into production. 
* Consider the infrastructure requirements for deploying the model, such as computing resources, storage, and network bandwidth. 
* Ensure that the model's performance meets the business requirements and monitor its performance regularly. 
* Develop a plan for maintaining and updating the model to ensure that it remains accurate and relevant over time.
* Consider ethical and legal implications of deploying the model and ensure that it meets ethical and regulatory guidelines.

14. **Monitoring**
* Once the model is deployed, it is important to monitor its performance over time. 
* Monitor the inputs and outputs to ensure that they are consistent with the training data. 
* Check for any data drift or concept drift, which occurs when the statistical properties of the input data change over time, leading to a decrease in the model's performance. 
* Use techniques such as statistical process control, anomaly detection, or change detection to identify any issues and take corrective actions.

15. **Maintenance**
* Maintain the model by periodically retraining it on new data or updating it with new features or hyperparameters.
* Consider the ongoing costs and benefits of maintaining the model compared to redeveloping it from scratch.
* Ensure that the model is still aligned with the business objectives and that the assumptions made during the development process are still valid.
* Document any changes or updates to the model and its associated processes and workflows.


## Unsupervised Learning
Placeholder

## Data Visualization
Placeholder

## Story Telling With Data
Placeholder

## Machine Learning For Production
Placeholder


# Deep Learning
Placeholder


# Software Engineering
## Algorithms
#### Advent Of Code 2021
https://github.com/yenh-cs/adventofcode2021.git

#### Advent Of Code 2022
https://github.com/yenh-cs/adventofcode2022.git

#### Leetcode
https://github.com/yenh-cs/algorithm.git 

## Python
#### 100 Days of Python
https://github.com/yenh-cs/100DaysPython.git




