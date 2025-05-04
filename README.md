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

* Preprocessed data using Pandas and Scikit-learn
* Scaled features using `StandardScaler`
* Used an ANN model built with Keras (TensorFlow backend):

  * Input Layer → Dense → Dropout → Dense → Output (Softmax)
  * Activation functions: ReLU for hidden layers, Softmax for output
  * Added dropout layers to reduce overfitting
* Loss function: `categorical_crossentropy`
* Optimizer: Adam
* Evaluation metrics: Accuracy, Loss

### Final Model Architecture

```
Dense (64 neurons, relu)
Dropout (rate=0.3)
Dense (32 neurons, relu)
Dropout (rate=0.3)
Output Dense Layer (softmax)
```

## ✅ Results

* **Test Loss:** 0.5862
* **Test Accuracy:** 71.81%

This result is promising considering the model simplicity and nature of the data. It suggests ANN models can detect some meaningful patterns in medical data.

## 📊 Visualization and Interpretability

I tried to understand the weights and activations of the model by:

* Extracting first layer weights
* Attempting to visualize neuron activations (faced technical issues)

This gave me better intuition about how features are treated inside the network.

## 📁 Files Included

* `epilepsy_ann_model.ipynb`: Full Google Colab notebook
* `epilepsy_dataset.csv`: Kaggle dataset
* `/images/neural_network_diagram.png`: Visual for ANN structure
* `README.md`: This file
