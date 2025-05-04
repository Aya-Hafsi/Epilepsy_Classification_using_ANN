# Epilepsy Type Classification using Artificial Neural Networks (ANN)

## 🔍 Project Overview

This project explores how Artificial Neural Networks (ANNs) can be used to predict the type of epilepsy a patient might have based on a set of clinical and lifestyle features.

### 🧠 Context

Epilepsy is a neurological condition characterized by recurrent seizures. The type of epilepsy can influence treatment plans and prognosis. Here, we attempt to build a model that classifies the type of epilepsy using non-invasive data, such as age, medical history, symptoms, and lifestyle factors.

## 🎯 Objective

Our main question was:

**"Can we help doctors or researchers predict the type of epilepsy a patient might have based on symptoms and lifestyle data?"**

By using an ANN on this structured dataset, we aimed to develop a simple but informative model that could offer insights into how machine learning models might assist in medical diagnostics.

## 📁 Dataset Description

The dataset contains 30 columns and represents various patient attributes, symptoms, and medical background details. Each row corresponds to one patient.

* Examples of input features:

  * Age, Gender, Weight, Height
  * Alcohol or Drug Use, Stress Before Episode, Family History of Epilepsy
  * Eye Rolling, Postictal Confusion, Seizure Duration, etc.

* Target:

  * `Target/Epilepsy Type` (multi-class): One-hot encoded into multiple columns such as:

    * `Target/Epilepsy Type_Complicated`
    * `Target/Epilepsy Type_Focal`
    * `Target/Epilepsy Type_Generalized`, etc.

No missing values were found in the dataset.

## 🧪 Methodology

To build this project, I followed a simple but structured pipeline combining data preprocessing, model building, and evaluation.

### 🔄 Data Preparation

* Loaded and explored the dataset using **Pandas**
* Encoded the multi-class target variable
* Scaled numerical features using **StandardScaler** from Scikit-learn

### 🧠 Neural Network Architecture

I used **Keras (TensorFlow backend)** to build a simple but effective ANN:

```plaintext
Input Layer
↓
Dense Layer (64 units, ReLU)
↓
Dropout Layer (rate = 0.3)
↓
Dense Layer (32 units, ReLU)
↓
Dropout Layer (rate = 0.3)
↓
Output Layer (Softmax)
```

* **Activation Functions**: ReLU (hidden), Softmax (output)
* **Loss Function**: `categorical_crossentropy`
* **Evaluation Metrics**: Accuracy and Loss

---

## 📊 Results

After training the model on a portion of the dataset and testing on unseen data:

* **Test Accuracy**: `71.81%`
* **Test Loss**: `0.5862`

This shows that even a relatively simple ANN can extract useful patterns from structured clinical data.

---

## ✅ Results

* **Test Loss:** 0.5862
* **Test Accuracy:** 71.81%

This result is promising considering the model simplicity and nature of the data. It suggests ANN models can detect some meaningful patterns in medical data.

## 📊 Visualization and Interpretability

I tried to understand the weights and activations of the model by:

* Extracting first layer weights
* Attempting to visualize neuron activations (faced technical issues)

This gave me better intuition about how features are treated inside the network.

## 📌 Reflections

* Learned how to apply an ANN to a real-world medical dataset
* Gained hands-on experience with preprocessing, modeling, and evaluating a classification task
* Would like to explore more advanced models in the future (e.g., deeper networks, hyperparameter tuning, etc.)

---

## 📁 Files

* `Epilepsy_Classification_using_ANN.ipynb`: Main Colab notebook
* `Epilepsy_dataset.csv`: Kaggle dataset
* `README.md`: Project overview and details

---

## Thanks for visiting! 😊 
If you found this project interesting, feel free to start the repo or connect with me on [LinkedIn](https://www.linkedin.com/in/aya-hafsi/)

