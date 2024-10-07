# Machine Learning - Sharif University of Technology
Solutions for assignments and project of ml course at Sharif university of Technology (CE-477)

# Table of Contents
> - [Heart_Disease_Prediction](#Heart_Disease_Prediction)
> - [MLE_MAP](#MLE_MAP)
> - [Polynomial_Regression](#Polynomial_Regression)
> - [Regularization](#Regularization)

   
* ## ***Heart_Disease_Prediction***
    In this assignment, I worked on [*Heart_Disease*](https://www.kaggle.com/johnsmith88/heart-disease-dataset) dataset.
    Topics I used:
    > * EDA
    > * Perceptron
    > * Naive Bayes
    
    Also these basic topics used:
    > * Confusion Matrix
    > * F1-score
    > * Recall_score
    > * Precision_score

    For reading the details quickly you can see [this pdf](https://github.com/Amirreza81/Machine-Learning/blob/main/Heart-Disease-Prediction/Heart_Disease_Prediction.pdf).<br _>
    For reading the notebook you can see this [link](https://github.com/Amirreza81/Machine-Learning/blob/main/Heart-Disease-Prediction/Heart_Disease_Prediction.ipynb).
    <br _>

* ## ***MLE_MAP***
    This exercise will help you gain a deeper understanding of, and insights into, Maximum Likelihood Estimation (*MLE*) and Maximum A Posteriori (*MAP*) estimation.
    <br _> For reading the details quickly you can see [this pdf](https://github.com/Amirreza81/Machine-Learning/blob/main/MLE-MAP/MLE_MAP.pdf).<br _>
    For reading the notebook you can see this [link](https://github.com/Amirreza81/Machine-Learning/blob/main/MLE-MAP/MLE_MAP.ipynb).


* ## ***Polynomial_Regression***
  This exercise explores polynomial regression, a form of regression analysis where the relationship
  between the independent variable ( X ) and the dependent variable ( y ) is modeled as an ( n )th
  degree polynomial. We will create a synthetic dataset, train models with varying degrees of
  polynomials, and evaluate their performance on different test sets.<br _>
  
  *Steps*:
  > * Create a synthetic dataset
  > * Splitting the Dataset
  > * Polynomial Regression Training
  > * Model Evaluation
  > * Plotting Model Scores

  For reading the details quickly you can see [this pdf](https://github.com/Amirreza81/Machine-Learning/blob/main/Polynomial-Regression/Polynomial_Regression.pdf).<br _>
  For reading the notebook you can see this [link](https://github.com/Amirreza81/Machine-Learning/blob/main/Polynomial-Regression/Polynomial_Regression.ipynb).
  <br _>

* ## ***Regularization***
    In this assignment, we will work with a dataset that includes The Boston housing data was collected in 1978 and each of the 506 entries
    represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. First, we will start by fitting a
    basic regression model using scikit-learn (sklearn) to establish a baseline for comparison. This basic regression model will serve as a reference
    point for evaluating the performance of more sophisticated models incorporating regularization techniques.
    Furthermore, we will apply L1 (Lasso) and L2 (Ridge) regularization techniques to refine our predictions and evaluate the impact of these
    methods on the accuracy of our results. <br _>

    *Topics*:
    > * L1 (Lasso) regularization
    > * L2 (Ridge) regularization

    For reading the details quickly you can see [this pdf](https://github.com/Amirreza81/Machine-Learning/blob/main/Regularization/Regularization.pdf).<br _>
    For reading the notebook you can see this [link](https://github.com/Amirreza81/Machine-Learning/blob/main/Regularization/Regularization.ipynb).
    <br _>

* ## ***KMeans***

    This notebook applies KMeans clustering on a dataset using both **Elbow Method** and **Silhouette Method** to determine the optimal number of clusters. The project compares the performance of a custom KMeans implementation with the one from Sklearn.

    ## Methods
    > * **Elbow Method**: Focuses on minimizing WCSS (within-cluster sum of squares) to identify the point where adding more clusters doesn't significantly improve results.
    > * **Silhouette Method**: Evaluates cluster quality by measuring how well points fit within their own cluster vs. other clusters. A higher silhouette score indicates better-defined clusters.

    ## Results
    - **Elbow Method**: Optimal number of clusters is suggested as 3 or 4.
    - **Silhouette Method**: Optimal number of clusters is 2 based on the highest silhouette scores.
    - Silhouette method is preferred due to its deterministic nature and higher precision.