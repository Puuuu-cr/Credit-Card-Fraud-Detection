# Credit Card Fraud Detection

Detecting credit card fraud using **Random Forest** and **Logistic Regression**. For the Logistic Regression model, **undersampling** and **class weighting** were applied to address the imbalanced dataset. The experiments also included **time-based feature engineering**, **log-transformation of the transaction amount**, and **stratified sampling with Stratified K-Fold Cross-Validation** to select optimal hyperparameters.

---

## 📈 Model Performance Comparison

Performance was evaluated using the **Precision-Recall (PR) Curve**, which is more informative than accuracy for imbalanced datasets.

| Model & Strategy | AUPRC |
| :--- | :--- |
| **Random Forest (Balanced Subsample)** | **0.8501** |
| Logistic Regression (Undersampling + Log Transform) | 0.7525 |
| Logistic Regression (Balanced Weighting + Normalization) | 0.7476 |

### 🔍 Insights
* **Random Forest** achieved the best performance (AUPRC 0.85) due to its ability to capture **non-linear relationships**, such as bimodal transaction amounts and temporal spikes in fraud.  
* For **Logistic Regression**, **undersampling** slightly outperformed class weighting, allowing the linear model to define a clearer decision boundary in sparse feature space.  
* Tree-based models are generally more robust for highly imbalanced and complex fraud datasets.

---

## 🛠️ Technical Stack

* **Data Processing**: Pandas, NumPy  
* **Visualization**: Matplotlib, Seaborn  
* **Machine Learning**: Scikit-learn (Logistic Regression, Random Forest)  
