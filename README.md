# Data Science in Plain Text

Welcome to 'Data Science in Plain Text', a page dedicated to making data science accessible to everyone. 

Data science has become an increasingly important field, with applications ranging from business intelligence to healthcare and beyond. However, with all of the technical jargon and complex algorithms, it can often seem intimidating and inaccessible to those without a background in programming or statistics. 

That's where this blog comes in. My goal is to break down the concepts and tools of data science into plain language, making it easier for anyone to understand and apply to their work or personal projects. Whether you're a beginner just starting out or a seasoned data professional looking to expand your knowledge, my articles will cover a range of topics, from data cleaning and visualization to machine learning and beyond. I believe that everyone should have access to the power of data science, and I hope that this blog will serve as a valuable resource for anyone looking to learn more about this exciting field. 

Thank you for joining me on this journey to demystify data science and make it more accessible to everyone.

&nbsp;&nbsp;&nbsp;
# Mathematics for ML
## Statistics 101
#### **Measures of Central Tendency and Data Dispersion**
The first stage of any machine learning project is to explore the data, and a crucial part of that exploration is understanding the measures of central tendency and dispersion. 

It's always a good idea to summarize the data by means of graphs.
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


&nbsp;
#### **Statistical Power**
Statistical power refers to the probability of correctly rejecting a null hypothesis when it is false. In other words, it measures the ability of a statistical test to detect an effect or relationship if it truly exists in the population being studied.

A high statistical power indicates a greater likelihood of finding a significant result if the effect or relationship being tested actually exists. Conversely, a low statistical power means there is a higher chance of failing to detect a true effect, leading to a false negative result (Type II error).

Statistical power is influenced by several factors, including the sample size, the magnitude of the effect being studied, the chosen significance level (typically denoted as alpha), and the variability in the data. Increasing the sample size and effect size, as well as reducing variability, generally increase statistical power.

Researchers often aim for a sufficiently high statistical power (e.g., typically 80% or higher) to ensure that their study has a good chance of detecting meaningful effects. Adequate statistical power is crucial for drawing reliable conclusions and avoiding false interpretations of the data.


&nbsp;
#### **The Perils of Multiple Comparison and P-hacking: Unveiling the Pitfalls of Data Analysis**
Data analysis plays a crucial role in scientific research and decision-making processes. However, there are certain pitfalls that researchers and analysts must be aware of to ensure the integrity and reliability of their findings. Two key challenges in data analysis are multiple comparison and p-hacking. In this article, we will explore these concepts, understand their implications, and provide detailed examples to illustrate their perils.

**Multiple Comparison**
Multiple comparison refers to the practice of conducting multiple statistical tests or hypothesis tests simultaneously without appropriate adjustments. When multiple tests are performed on the same dataset or multiple variables are examined, the likelihood of obtaining false positive results increases. This is because random chance alone can lead to the appearance of significant findings, even if there is no true effect or relationship in the population.

Example: Imagine a medical study investigating the effects of a new drug on various health outcomes. Researchers decide to compare the drug's impact on blood pressure, cholesterol levels, and glucose levels. If each outcome is analyzed independently without adjusting for multiple comparisons, there is a higher chance of observing significant results purely due to chance, leading to potential false conclusions.

**P-hacking**
P-hacking, also known as data dredging or fishing, involves manipulating data or analysis procedures to obtain statistically significant results. It refers to the selective reporting or analysis of data in order to achieve desired outcomes or significant p-values. This practice is highly problematic because it can introduce bias, mislead interpretations, and undermine the credibility of research findings.

Example: Suppose a social scientist is examining the relationship between social media usage and self-esteem. Initially, the researcher measures self-esteem in various ways and tests multiple statistical models. After several unsuccessful attempts to find a significant relationship, they finally discover a small but statistically significant correlation between self-esteem and the number of followers on a specific social media platform. However, this result was obtained after trying various combinations and selectively reporting the analysis that yielded a significant finding, thus introducing a bias in the reported results.

Both multiple comparison and p-hacking can have severe consequences. They can lead to spurious or false-positive findings, misinterpretations, and misguided decision-making. Moreover, they erode the trust in scientific research and can have far-reaching implications in fields such as medicine, economics, and social sciences.

**Mitigating the Pitfalls**
To mitigate the perils of multiple comparison and p-hacking, it is important to adopt responsible and rigorous practices in data analysis. Here are some key recommendations:

1. *Pre-registration*: Clearly define hypotheses and analysis plans in advance, reducing the temptation to selectively report findings post-analysis.

1. *Adjusting for Multiple Comparisons*: Utilize appropriate statistical techniques, such as Bonferroni correction, false discovery rate (FDR), or the family-wise error rate (FWER) adjustment, to account for the increased probability of false positives when conducting multiple tests.

1. *Transparency and Reproducibility*: Share data, analysis scripts, and methodologies to allow for scrutiny and replication by others, promoting transparency and reducing the potential for p-hacking.

1. *Independent Validation*: Encourage independent replication of findings to validate and reinforce the robustness of research outcomes.

Multiple comparison and p-hacking pose significant challenges to the integrity and credibility of data analysis. Researchers, analysts, and decision-makers must be aware of these pitfalls and adopt responsible practices to ensure rigorous and reliable results. By adhering to proper statistical methods, transparency, and robust scientific practices, we can enhance the validity of research findings, foster trust in data-driven decisions, and advance our understanding of the world around us.


&nbsp;
#### **Simpson's Paradox**
Simpson's Paradox is a statistical phenomenon where the relationship between two variables reverses or changes direction when a third variable is taken into account. It can lead to misleading or contradictory conclusions if the underlying confounding factor is not properly considered. Here are a few real-life examples to illustrate the concept:

1. Example 1: University Admissions Suppose a university is evaluating admission rates for different departments. When looking at each department individually, it appears that men have higher admission rates than women. However, when considering the overall data for the university as a whole, it is found that women have a higher admission rate. This paradox can occur if there are more women applying to departments with higher admission rates, while men tend to apply to departments with lower admission rates. The gender distribution across departments acts as a confounding factor, leading to the reversal of the relationship.

1. Example 2: Medical Treatment In a medical study comparing the effectiveness of two treatments, Treatment A and Treatment B, it is found that Treatment A has a higher success rate when analyzed separately for younger and older patients. However, when combining the data for all age groups, it is observed that Treatment B has a higher success rate overall. This paradox can arise if the distribution of age groups is not taken into account. For example, if Treatment A is more commonly administered to younger patients who generally have better outcomes, while Treatment B is used for older patients with more severe cases, the overall analysis may lead to a different conclusion.

1. Example 3: Sports Performance In sports, an athlete's performance can be evaluated based on various factors, such as individual game statistics or overall season performance. Consider a baseball player who has a higher batting average in each game against left-handed pitchers and right-handed pitchers separately. However, when looking at the combined data, it is found that the player has a lower overall batting average. This paradox can occur if the player faces left-handed pitchers more frequently in games where the overall team's performance is weaker, while facing right-handed pitchers more often in games where the team performs better. The team's performance acts as a confounding factor, leading to the reversal of the relationship.

These examples demonstrate how Simpson's Paradox can occur in real-life situations, where the interpretation of data can be misleading if the presence of confounding factors is not considered. It highlights the importance of thorough analysis and understanding the underlying factors to draw accurate conclusions from data.



&nbsp;&nbsp;
## Linear Algebra
Placeholder


&nbsp;&nbsp;
## Multivariate Calculus
Placeholder


&nbsp;&nbsp;&nbsp;
# Machine Learning
## Data Mining
#### **Cross Industry Standard Process for Data Mining (CRISP-DM)** - Genralist friendly version
![This is the Cross Industry Standard Process for Data Mining diagram.](/assets/images/CRISP-DM.png "This is the Cross Industry Standard Process for Data Mining diagram.")

Data mining is an intricate process that combines science, technology, and art to solve complex problems. This process follows a structured approach to ensure consistency, repeatability, and objectivity. It involves several stages that are iterative, meaning that they may need to be repeated until the problem is solved.

The first stage is **business understanding**, where analysts need to understand the problem that needs to be solved. This stage requires creativity as the business problem needs to be cast as one or more data science problems. Data scientists need to have high-level knowledge to see novel formulations.

The next stage is **data understanding**, where analysts need to understand the available raw material or data to build a solution. Data may have strengths and limitations, and it's essential to estimate the costs and benefits of each data source. Data collation may also require additional effort.

**Data preparation** is another crucial stage where data are manipulated and converted into forms that yield better results. Data mining techniques have certain requirements that may require data to be in a different form than how it is naturally provided. For example, converting data to tabular format, removing or inferring missing values, and converting data to different types. During data preparation, it's essential to beware of "leaks" that may give information on the target variable that's not actually available when the decision needs to be made.

The **modeling** stage is where data mining techniques are applied to the data to capture regularities in the data and develop models based on algorithms such as decision trees, neural networks, or association rules. Different algorithms may be better suited for different types of data and applications.

The **evaluation** stage is crucial to assess data mining results rigorously and gain confidence that they are valid and reliable. Models need to be tested in a controlled laboratory setting before deployment. Various stakeholders have interests in the business decision-making that will be accomplished or supported by the resultant models, and they need to sign off on the deployment of the models.

The **deployment** stage refers to integrating a model or solution into a production environment. One important consideration is that the environment it will operate in may be different from the environment it was developed in, leading to discrepancies in performance. Ongoing monitoring and maintenance are also necessary to ensure continued accuracy, as models can suffer from drift as data patterns change over time. It's crucial to evaluate the ethical implications of deploying a model or solution carefully.

For example, suppose a model is trained on data collected from a specific region to predict the likelihood of loan defaults. In that case, it may not perform as well when deployed in a different region due to variations in demographics and economic conditions. Also, suppose the model is used to make loan approval decisions, and it's found to be biased against a particular race or gender. In that case, it's important to evaluate the potential impact of the model before deploying it in a production environment. The model's behavior needs to be made comprehensible to stakeholders, and the ethical implications of its deployment should be carefully considered.


&nbsp;
#### **Cross Industry Standard Process for Data Mining (CRISP-DM)** - Detailed version
![This is the Cross Industry Standard Process for Data Mining diagram.](/assets/images/CRISP-DM.png "This is the Cross Industry Standard Process for Data Mining diagram.")

Data mining is a complex process that involves the application of science, technology, and art. There is a well-understood process that places a structure on the problem, allowing for consistency, repeatability, and objectiveness. The process diagram shows that iteration is the rule rather than the exception, as going through the process once without solving the problem is generally not a failure. The steps involved in data mining are discussed below:


**Business Understanding**
The first step in data mining is to understand the problem that needs to be solved. Often, business projects are not clear and unambiguous data mining problems, so recasting the problem and designing a solution is an iterative process of discovery. The business understanding stage involves creativity from the analysts, as they need to cast the business problem as one or more data science problems. High-level knowledge of the fundamentals helps creative business analysts see novel formulations.


**Data Understanding**
The data comprise the available raw material from which the solution will be built. It is important to understand the strengths and limitations of the data because rarely is there an exact match with the problem. The costs of data can vary, and a critical part of the data understanding phase is estimating the costs and benefits of each data source and deciding whether further investment is merited. Even after all datasets are acquired, collating them may require additional effort.


**Data Preparation**
Data preparation is a crucial stage of the data mining process where data are manipulated and converted into forms that yield better results. Data mining techniques impose certain requirements on the data they use, and often require data to be in a form different from how the data are provided naturally. Typical examples of data preparation include converting data to tabular format, removing or inferring missing values, and converting data to different types.

Some data mining techniques are designed for symbolic and categorical data, while others handle only numeric values. In addition, numerical values must often be normalized or scaled so that they are comparable. Standard techniques and rules of thumb are available for doing such conversions.

One very general and important concern during data preparation is to beware of "leaks." A leak is a situation where a variable collected in historical data gives information on the target variable, information that appears in historical data but is not actually available when the decision has to be made. Leakage must be considered carefully during data preparation because it typically is performed after the fact—from historical data.


**Modeling**
The modeling stage is the primary place where data mining techniques are applied to the data. It is important to have some understanding of the fundamental ideas of data mining, including the sorts of techniques and algorithms that exist, because this is the part of the craft where the most science and technology can be brought to bear.

The output of modeling is some sort of model or pattern capturing regularities in the data. Models can be based on many types of algorithms, such as decision trees, neural networks, or association rules. Each algorithm has strengths and weaknesses, and different algorithms may be better suited for different types of data and applications.


**Evaluation**
The purpose of the evaluation stage is to assess the data mining results rigorously and to gain confidence that they are valid and reliable before moving on. It is possible to deploy results immediately after data mining, but this is inadvisable; it is usually far easier, cheaper, quicker, and safer to test a model first in a controlled laboratory setting.

The evaluation stage also serves to help ensure that the model satisfies the original business goals. Usually, a data mining solution is only a piece of the larger solution, and it needs to be evaluated as such. Evaluating the results of data mining includes both quantitative and qualitative assessments. Various stakeholders have interests in the business decision-making that will be accomplished or supported by the resultant models.

In many cases, these stakeholders need to "sign off" on the deployment of the models and need to be satisfied by the quality of the model's decisions. To facilitate such qualitative assessment, the data scientist must think about the comprehensibility of the model to stakeholders (not just to the data scientists). If the model itself is not comprehensible (e.g., maybe the model is a very complex mathematical formula), the data scientists need to work to make the behavior of the model comprehensible.


**Deployment**
Deployment refers to the process of integrating a model or solution into a production environment. This can involve putting a predictive model into operation within an information system, or deploying a data-driven decision making process into a business operation.

One important consideration in deploying a model is that the environment it will operate in may be different from the environment it was developed in. This can lead to discrepancies in performance, and the need for recalibration or adjustment of the model. For example, a model trained on data collected from a specific region may not perform as well when deployed in a different region. Another important aspect of deployment is ongoing monitoring and maintenance. Models can suffer from "drift" as data patterns change over time, and periodic recalibration or retraining may be necessary to ensure continued accuracy.

Finally, it is important to consider the ethical implications of deploying a model or solution. Data-driven decision making can have unintended consequences, and it is important to carefully evaluate the potential impact of a model before deploying it in a production environment.


&nbsp;
#### **Data Science Basic Concepts**
##### **Signal and Noise**
Signal refers to the true underlying pattern or information in the data that we are interested in, while noise represents random variations or irrelevant factors that can obscure the signal. For instance, in analyzing stock market data, the signal may be the long-term trend indicating the overall performance, while the noise could be short-term fluctuations caused by random market events.


&nbsp;
##### **Overfitting**
Overfitting is a phenomenon in machine learning and statistical modeling where a model becomes overly complex and excessively tailored to the training data, resulting in poor generalization to new, unseen data. It occurs when a model captures noise or random fluctuations in the training data, instead of learning the underlying true patterns.

Here are some explained examples to help understand overfitting:

**Polynomial Regression**: Suppose you have a dataset with a single input variable (e.g., housing prices) and a target variable (e.g., house size). If you fit a high-degree polynomial regression model to the data, it may perfectly fit all the training examples, including the noise or outliers. The model will have numerous oscillations and wiggles to accommodate each training point. However, when you use this overfitted model to predict house sizes for new data points, it will likely produce unreliable and inaccurate results because it has memorized the noise rather than capturing the underlying trend.

**Decision Trees**: Decision trees are prone to overfitting when they become too deep and complex. Imagine you're building a decision tree to classify whether an email is spam or not based on various features. If the tree becomes too deep and branches too extensively, it can create specific rules for each training example, even those that are outliers or noise. Consequently, the tree may have high accuracy on the training set, but it will struggle to generalize well to new emails and may misclassify them.

**Neural Networks**: Neural networks, particularly deep networks with a large number of layers and parameters, are susceptible to overfitting. If you have a complex neural network architecture and insufficient training data, the model may effectively memorize the training examples, including the noise, rather than learning meaningful patterns. As a result, the network will struggle to make accurate predictions on unseen data.

**Image Classification**: In the context of image classification, overfitting can occur when a model is trained on a limited number of images. If the model becomes too complex, it may start memorizing specific details, textures, or backgrounds of the training images, rather than learning generalizable features. Consequently, it will struggle to classify new images correctly, especially those with variations in lighting, angles, or backgrounds.

To mitigate overfitting, various techniques can be employed, such as:

1. *Regularization*: Adding regularization terms to the model's loss function, like L1 or L2 regularization, penalizes overly complex models and encourages simplicity.
1. *Cross-validation*: Splitting the data into multiple subsets for training and validation helps assess the model's performance on unseen data and identify signs of overfitting.
1. *Early stopping*: Monitoring the model's performance during training and stopping the training process when the validation error starts increasing can prevent overfitting.
1. *Data augmentation*: Increasing the size of the training dataset through techniques like image transformations or synthetic data generation can help the model generalize better.
1. *Simplifying the model*: Using simpler models with fewer parameters or reducing the complexity of the model architecture can reduce the risk of overfitting.
By employing these techniques, the aim is to find the right balance between model complexity and generalization, ensuring that the model performs well not only on the training data but also on unseen data.

&nbsp;
##### **Under fitting**
Underfitting is the opposite of overfitting and occurs when a machine learning model or statistical model is too simple to capture the underlying patterns in the data. It arises when the model is unable to learn the complexities and nuances present in the data, resulting in poor performance on both the training data and new, unseen data.

Here are some explained examples to illustrate underfitting:

**Linear Regression**: Consider a scenario where you have a dataset with a single input variable and a target variable that doesn't have a linear relationship. If you fit a simple linear regression model to this data, it may have high bias and fail to capture the nonlinear relationship between the variables. The resulting line will be too rigid and unable to adequately represent the data points, leading to a poor fit.

**Classification**: Let's say you have a binary classification problem with two classes that are not linearly separable. If you attempt to use a linear classifier, such as logistic regression, it may draw a straight line to separate the classes. However, this linear decision boundary will not be able to accurately classify the data, resulting in a high error rate.

**Underfitting in Neural Networks**: In the context of neural networks, underfitting can occur when the network is not deep or wide enough to capture the complexity of the underlying data. If you have a complex problem, but use a shallow neural network with only a few layers and a limited number of neurons, it may struggle to learn the intricate patterns, leading to poor performance.

**Image Recognition**: Suppose you are training a deep convolutional neural network (CNN) for image recognition tasks, but you provide the network with limited training data and a relatively simple architecture. The model may fail to capture the intricate features and details in the images, resulting in low accuracy and an inability to generalize well to new, unseen images.

To mitigate underfitting, several approaches can be employed:

1. *Increasing model complexity*: Using more complex models with greater capacity, such as deep neural networks or nonlinear models, can help capture the underlying patterns in the data.
1. *Feature engineering*: Enhancing the dataset by incorporating additional relevant features or transforming the existing features can provide the model with more discriminatory information.
1. *Collecting more data*: Gathering a larger and more diverse dataset can help the model learn the underlying patterns better and reduce the risk of underfitting.
1. *Reducing regularization*: If the model is overly regularized, loosening the regularization constraints, such as decreasing the strength of regularization terms, can allow the model to learn more complex relationships in the data.
1. *Hyperparameter tuning*: Adjusting the hyperparameters of the model, such as learning rate, number of hidden units, or depth of the network, can improve the model's capacity to capture the underlying patterns.
The goal in mitigating underfitting is to find an appropriate level of model complexity that balances simplicity with the ability to capture the relevant patterns in the data, ultimately leading to better generalization performance.Underfitting is the opposite of overfitting and occurs when a machine learning model or statistical model is too simple to capture the underlying patterns in the data. It arises when the model is unable to learn the complexities and nuances present in the data, resulting in poor performance on both the training data and new, unseen data.



&nbsp;
##### **Bias-Variance Tradeoff**
The bias-variance tradeoff is a fundamental concept in machine learning that helps us understand the relationship between the complexity of a model and its ability to generalize well to unseen data. Let's break it down with a simple example.

Imagine we're training a model to predict house prices based on their size (in square meter). We have a dataset of houses with their corresponding sizes and prices. Now, we want to build a regression model to predict the price of a new house given its size.

**High Bias, Low Variance**
If we use a simple model, such as a linear regression with only one feature (size), it may not capture the complexity of the underlying relationship accurately. This is called high bias. The model assumes a linear relationship between size and price, which may not hold true in real-world scenarios. The model's predictions might be consistently off the mark, showing high error on both the training and test data. This is an example of high bias and low variance.

**Low Bias, High Variance**
On the other hand, if we use a more complex model, like a high-degree polynomial regression, it can fit the training data very well. The model can flexibly capture any pattern, even if it's noisy or random, resulting in low bias. However, when we evaluate the model on the test data, it may perform poorly. This is because the model has overfit the training data, learning the noise and idiosyncrasies of the training set. The model is too sensitive to small changes in the data, leading to high variance. This means the model may not generalize well to new, unseen houses.

So, we have a tradeoff between bias and variance:

**Bias**: Measures the error introduced by the model's simplifying assumptions or incorrect assumptions about the relationship between features and the target variable. High bias can lead to underfitting.

**Variance**: Measures the model's sensitivity to fluctuations or noise in the training data. High variance can lead to overfitting.
The goal is to find an optimal balance between bias and variance, which results in good generalization performance on unseen data. This is known as the sweet spot of the bias-variance tradeoff. It's about finding a model that is complex enough to capture the underlying patterns but not so complex that it starts memorizing noise or idiosyncrasies of the training data.

In practice, this tradeoff is managed through techniques like regularization, feature selection, and model selection. Regularization helps reduce variance by imposing constraints on the model's complexity. Feature selection helps reduce bias by including relevant features. Model selection involves choosing a model that provides the right amount of complexity for the given problem.

By understanding and balancing the bias-variance tradeoff, we can develop models that generalize well, avoiding underfitting (high bias) and overfitting (high variance).


&nbsp;&nbsp;
## Supervised Learning
#### **A Comprehensive Template For Classification Problem**
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


&nbsp;
#### **A Comprehensive Template For Regression Problem**
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


&nbsp;&nbsp;
## Unsupervised Learning
Placeholder


&nbsp;&nbsp;
## Data Visualization
Placeholder


&nbsp;&nbsp;
## Story Telling With Data
Placeholder


&nbsp;&nbsp;
## Machine Learning For Production
Placeholder


&nbsp;&nbsp;&nbsp;
# Deep Learning
## Neural Networks
Placeholder


&nbsp;&nbsp;
## Convolutional Neural Networks
Placeholder


&nbsp;&nbsp;
## Keras and TensorFlow
Placeholder



&nbsp;&nbsp;&nbsp;
# Software Engineering
## Algorithms
#### Advent Of Code 2021
https://github.com/yenh-cs/adventofcode2021.git

&nbsp;
#### Advent Of Code 2022
https://github.com/yenh-cs/adventofcode2022.git

&nbsp;
#### Leetcode
https://github.com/yenh-cs/algorithm.git 


&nbsp;&nbsp;
## Python
#### 100 Days of Python
https://github.com/yenh-cs/100DaysPython.git




