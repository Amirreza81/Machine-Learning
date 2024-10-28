# Machine Learning
Solutions for assignments and project of machine learning course at **Sharif university of Technology** (CE-477)

# Table of Contents
> - [Heart_Disease_Prediction](#Heart_Disease_Prediction)
> - [MLE_MAP](#MLE_MAP)
> - [Polynomial_Regression](#Polynomial_Regression)
> - [Regularization](#Regularization)
> - [KMeans](#KMeans)
> - [Classification using PyTorch](#Bank_Marketing_Classification_using_PyTorch)
> - [Dimension Reduction](#Dimension_Reduction)
> - [KNN from scratch](#KNN)
> - [SVM](#SVM)

   
* # ***Heart_Disease_Prediction***
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

* # ***MLE_MAP***
    This exercise will help you gain a deeper understanding of, and insights into, Maximum Likelihood Estimation (*MLE*) and Maximum A Posteriori (*MAP*) estimation.
    <br _> For reading the details quickly you can see [this pdf](https://github.com/Amirreza81/Machine-Learning/blob/main/MLE-MAP/MLE_MAP.pdf).<br _>
    For reading the notebook you can see this [link](https://github.com/Amirreza81/Machine-Learning/blob/main/MLE-MAP/MLE_MAP.ipynb).


* # ***Polynomial_Regression***
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

* # ***Regularization***
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

* # ***KMeans***

    This notebook applies KMeans clustering on a dataset using both **Elbow Method** and **Silhouette Method** to determine the optimal number of clusters. The project compares the performance of a custom KMeans implementation with the one from Sklearn.

    Methods:
    > * **Elbow Method**: Focuses on minimizing WCSS (within-cluster sum of squares) to identify the point where adding more clusters doesn't significantly improve results.
    > * **Silhouette Method**: Evaluates cluster quality by measuring how well points fit within their own cluster vs. other clusters. A higher silhouette score indicates better-defined clusters.

    Results:
    - **Elbow Method**: Optimal number of clusters is suggested as 3 or 4.
    - **Silhouette Method**: Optimal number of clusters is 2 based on the highest silhouette scores.
    - Silhouette method is preferred due to its deterministic nature and higher precision.

    For reading the notebook you can see this [link](https://github.com/Amirreza81/Machine-Learning/blob/main/KMeans/kmeans.ipynb).

* # ***Bank_Marketing_Classification_using_PyTorch***

    This notebook is focused on performing a classification task on a bank marketing dataset using PyTorch. Below is a summary of the key steps and components of the notebook.

    ### 1. Importing Libraries
    The following libraries are used for data manipulation, machine learning, and neural network construction:
    - **PyTorch**: `torch`, `torch.nn`, `torch.optim`
    - **Data Processing**: `pandas`, `numpy`, `sklearn`
    - **Visualization**: `matplotlib`
    - **Data Handling**: `TensorDataset`, `DataLoader`

    ### 2. Loading and Preprocessing the Data
    ### Dataset
    The dataset used is the **Bank Marketing Dataset**, which is loaded from a CSV file. It contains various features that describe customer information and whether they subscribed to a bank product.

    ### Steps:
    - **Train-Test Split**: The data is split into train, validation, and test sets using `train_test_split` from `sklearn`.
    - **Feature Scaling**: Continuous variables such as age, balance, and duration are normalized using `StandardScaler`.
    - **Encoding Categorical Variables**: One-hot encoding is applied to categorical features (job, marital status, etc.) using `pandas.get_dummies`, and the target label (`y`) is encoded using `LabelEncoder`.

    For reading the notebook you can see this [link](https://github.com/Amirreza81/Machine-Learning/blob/main/Classification%20using%20PyTorch/ML_HW4_Answer.ipynb).


* # ***Dimension_Reduction***

    This project demonstrates various techniques for dimensionality reduction applied to a dataset. The goal is to reduce the number of dimensions while preserving as much relevant information as possible. This can help in visualization and improving the performance of machine learning models.

    ### Methods Used

    1. **Data Preprocessing**
    - The dataset (`nutrition.csv`) is loaded using `Pandas`.
    - Only numeric columns are selected for analysis.
    - Data is scaled using `StandardScaler` to normalize the feature values.

    2. **Dimensionality Reduction Techniques**
    - **PCA (Principal Component Analysis)**: This method reduces the dimensionality by transforming the original variables into a smaller set of new variables (principal components), which capture the       most variance.
    - **ICA (Independent Component Analysis)**: A technique that focuses on making the components as statistically independent as possible, useful for separating mixed signals.
    - **t-SNE (t-distributed Stochastic Neighbor Embedding)**: A non-linear technique mainly used for the visualization of high-dimensional data. It maps multi-dimensional data to a two or three-              dimensional space.

    3. **Visualization**
    - Visualizations are generated using `matplotlib` and `seaborn` to compare results and understand the structure of the data after applying each technique.

    For reading the notebook you can see this [link](https://github.com/Amirreza81/Machine-Learning/blob/main/Dimension-Reduction/dimension_reduction.ipynb).


* # ***KNN***

    This notebook demonstrates a complete, step-by-step implementation of the K-Nearest Neighbors (KNN) algorithm from scratch. It covers key concepts, code implementation, and model evaluation.

    ## Objectives
    - Understand the basics of the KNN algorithm.
    - Implement the KNN algorithm without using specialized libraries.
    - Evaluate the model's performance on test data.
    - Visualize the data and results to understand KNN behavior.

    ## Steps

    ### 1. Import Libraries
    ### 2. Define Distance Function (e.g., Euclidean)
    ### 3. Implement KNN Function from Scratch
    ### 4. Data Preparation
    ### 5. Model Evaluation
    ### 6. Data and Results Visualization


    For reading the notebook you can see this [link](https://github.com/Amirreza81/Machine-Learning/blob/main/KNN/knn-from-scratch-Answer.ipynb).
    For reading the complete README you can see [here](https://github.com/Amirreza81/Machine-Learning/blob/main/KNN/readme.md).


* # ***SVM***

    In this assignment, I implemented SVM (Support Vector Machines) for classification.

    ### Steps:
    - Data Preprocessing
    - Model 
    - Evaluation 
    - Fine-tuning
    - Multiclass SVM
    - Different SVM Kernels:
        - Linear Kernel
        - Gaussian RBF Kernel
        - Polynomial Kernel
        - Sigmoid Kernel

    For reading the notebook you can see this [link](https://github.com/Amirreza81/Machine-Learning/blob/main/SVM/SVM.ipynb).