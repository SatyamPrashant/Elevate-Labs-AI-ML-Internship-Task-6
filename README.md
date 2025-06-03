# Elevate Labs AI/ML Internship Task 6: K-Nearest Neighbors (KNN) – Iris Classification

---

## Dataset
- **Source:** [Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)  
- **Shape:** 150 samples × 5 columns  
- **Features:**  
  - `SepalLengthCm`  
  - `SepalWidthCm`  
  - `PetalLengthCm`  
  - `PetalWidthCm`  
- **Target:** `Species` (three classes:  
  `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`)

---

## Libraries / Tools
- Python 3.x  
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  

---

## Steps Performed

### 1. Data Loading & Inspection
- Loaded `Iris.csv` into a Pandas DataFrame.  
- Dropped the `Id` column (not relevant to modeling).  
- Verified no missing values and checked class distribution.

### 2. Feature Preparation & Normalization
- Separated features (`X`) and target (`y`).  
- Standardized features using `StandardScaler` (mean=0, std=1).  
- Converted scaled features back to DataFrame for clarity.

### 3. Train/Test Split
- Performed an **80% train / 20% test** split with `stratify=y` to preserve class proportions.  
- Confirmed shapes: 
  - `X_train_scaled`: (120, 4), `X_test_scaled`: (30, 4).

### 4. Finding the Optimal K
- Evaluated K values from 1 to 15.  
- Recorded both training and test accuracy for each K.  
- Plotted **Accuracy vs. K** and selected the K with highest test accuracy (K = 9).

### 5. Final Model Evaluation (K = 9)
- Trained `KNeighborsClassifier(n_neighbors=9)` on training data.  
- Predicted on the test set.  
- **Test Accuracy:** 0.9667 (96.67%)  

  
### 6. Decision Boundary Visualization
- Plotted decision regions using only the first two features (`SepalLengthCm`, `SepalWidthCm`).  
- Overlaid training points to illustrate how KNN partitions the feature space.  

### 7. Model Persistence
- Saved the final model as `knn_iris_model.pkl`.  
- Saved the `StandardScaler` as `iris_scaler.pkl` for future normalization of new data.

---

## Interview Insights

- **Q: How does KNN work?**  
A: KNN classifies a sample based on majority votes from its K nearest neighbors in feature space.

- **Q: How to choose the right K?**  
A: Plot train vs. test accuracy for different K values; select K with highest test accuracy to balance bias and variance.

- **Q: Why normalize features in KNN?**  
A: Distance-based algorithms are sensitive to scale; without scaling, features with larger ranges dominate distance calculations.

- **Q: Is KNN sensitive to noise?**  
A: Yes; using a larger K can help smooth decision boundaries and reduce noise sensitivity.

---

## How to Reproduce

1. **Clone this repository.**  
2. Ensure `Iris.csv` is in the root directory.  
3. Open `task6_knn_iris_classification.ipynb` in Jupyter Notebook or JupyterLab.  
4. Run each cell in sequence:  
 - Data loading & inspection  
 - Feature normalization  
 - Train/test split  
 - K vs. accuracy analysis  
 - Final model training & evaluation  
 - Decision boundary plotting  
 - Model saving  

---
