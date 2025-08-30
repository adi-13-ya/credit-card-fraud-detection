# 🚨 Credit Card Fraud Detection with Machine Learning

This repository demonstrates how to train, evaluate, and analyze a fraud detection model using the **Kaggle Credit Card Fraud dataset**.  
The project walks through **data preprocessing, model training, and evaluation** of fraud detection techniques on an imbalanced dataset.

---

## 📂 Dataset

We used the Kaggle dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- Transactions made by credit cards in **September 2013 by European cardholders**.  
- **Total Transactions:** 284,807  
- **Fraudulent Transactions:** 492 (≈0.172%) → Highly imbalanced dataset  
- **Time**: Seconds elapsed since the first transaction in the dataset  
- **Amount**: Transaction amount (useful for cost-sensitive learning)  
- **Features V1–V28**: PCA-transformed features (due to confidentiality)  
- **Class**: Target variable → `1 = Fraud`, `0 = Legitimate`

---

## 🔄 Workflow

### 1. **Data Preprocessing**  
   - Handling **imbalanced dataset** (fraud ratio kept consistent).  
   - Feature **classification & standardization** using `StandardScaler`.  
   - Standardization improves **neural network convergence**.  
   - **Train-test split** with `stratify` to preserve fraud ratio.

   📌 Output:
   
   *Data prepared. Training samples: 227845 Test samples: 56962*  

### 2. **Model Training**  
We trained multiple models including Neural Networks:  

- **Neural Network Architecture**:  
  - Hidden layers with **ReLU activation**  
  - Output layer with **Sigmoid activation** (for binary classification)  
- Optimizer: **Adam**  
- Early stopping to prevent overfitting  

📌 Output
   
Model Training by running *model.fit()* function

 ![*model-training*](/images/image-1.png) 

### 3. **Evaluation**  
We evaluated using:  

-  Confusion Matrix  
-  ROC Curve & AUC Score  
-  Classification Report  

*Confusion Matrix*
![confusion-matrix](/images/image-2.png)   

*ROC Curve*
![roc-curve](/images/image-3.png)
---

## 🛠️ Tech Stack  

- Python   
- Pandas, NumPy → Data preprocessing  
- Scikit-learn → ML utilities (StandardScaler, train_test_split, metrics)  
- TensorFlow / Keras → Neural Networks  
- Matplotlib, Seaborn → Visualizations  

---

## 📈 Results  

- Test Accuracy: 97.83%  
- AUC Score: 0.9578  
- Fraud Recall: 86.73%  
- Test Loss: 0.0575


---

## 🚀 How to Run This Project

### 1. Clone the Repository
```bash
git clone https://github.com/adi-13-ya/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Data File

Download the dataset from **Kaggle**:  
[*Credit Card Fraud Detection Dataset*](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

1. Login to your Kaggle account (create one if you don’t have it).  
2. Download the dataset file.  
3. If you receive a `.zip` file, extract it to get the `.csv`.  
4. Rename the file to `creditcard.csv`.  
5. Place the file inside the `/data` folder of this project.  



### 4. Run the code
```bash
jupyter notebook src/train_model.ipynb
```
---

## 🔮 Future Improvements

- ✅ Experiment with advanced deep learning models like LSTMs, Autoencoders, or Ensemble methods

- ✅ Implement SMOTE / Oversampling / Undersampling for better handling of imbalanced data

- ✅ Perform hyperparameter optimization using GridSearch or Bayesian Optimization

- ✅ Deploy the model using Flask, FastAPI, or Streamlit for real-time fraud detection

- ✅ Integrate with cloud platforms (AWS, GCP, Azure) for scalability

---
## 👥 Authors  
- **Aarav Jha** [🔗](https://github.com/aaravjha77)
- **Aditya Gujar** [🔗](https://github.com/adi-13-ya) 

---

## 📜 License  
This project is licensed under the **MIT License** – you are free to use, modify, and distribute this project with proper attribution.  

