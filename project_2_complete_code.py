#!/usr/bin/env python3
"""
PROJECT 2: Data Classification Using AI
DecodeLabs Batch 2026
Algorithm: K-Nearest Neighbors (KNN)
Dataset: Iris Flowers
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# COLOR CODES FOR TERMINAL OUTPUT
# ================================================================
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    ORANGE = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    END = '\033[0m'

def print_section(title):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.END}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.CYAN}ℹ️  {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.ORANGE}⚠️  {msg}{Colors.END}")

# ================================================================
# STEP 1: LOAD AND EXPLORE DATASET
# ================================================================
print_section("STEP 1: LOAD & EXPLORE DATASET")

iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Create DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['species'] = df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

print_info(f"Dataset Shape: {df.shape}")
print_info(f"Features: {iris.feature_names}")
print_info(f"Classes: {iris.target_names}")
print("\nDataset Preview:")
print(df.head(10))
print("\nClass Distribution:")
print(df['species'].value_counts())
print("\nStatistical Summary:")
print(df.describe())

# ================================================================
# STEP 2: FEATURE SCALING (The Gatekeeper Rule)
# ================================================================
print_section("STEP 2: FEATURE SCALING")

print_warning("KNN uses distance-based calculations. Scaling is CRITICAL!")
print_info("Without scaling: Features with larger ranges dominate distance")
print_info("With scaling: All features contribute equally")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print_success("Features scaled using StandardScaler")
print_info(f"Raw data range: [{X.min():.2f}, {X.max():.2f}]")
print_info(f"Scaled data range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
print_info(f"Scaled mean (should be ~0): {X_scaled.mean(axis=0)}")
print_info(f"Scaled variance (should be ~1): {X_scaled.var(axis=0)}")

# ================================================================
# STEP 3: TRAIN-TEST SPLIT (Structural Integrity)
# ================================================================
print_section("STEP 3: TRAIN-TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducibility
    shuffle=True        # Remove order bias
)

print_success("Data split completed")
print_info(f"Training set: {X_train.shape[0]} samples (80%)")
print_info(f"Testing set: {X_test.shape[0]} samples (20%)")
print_info(f"Training class distribution: {np.bincount(y_train)}")
print_info(f"Testing class distribution: {np.bincount(y_test)}")

# ================================================================
# STEP 4: FIND OPTIMAL K VALUE (Tuning the Engine)
# ================================================================
print_section("STEP 4: FIND OPTIMAL K VALUE")

print_warning("K=1: May overfit (too sensitive to noise)")
print_warning("K=150: May underfit (too generic)")
print_info("Finding the 'elbow point' for optimal K...")

error_rates = []
accuracies = []
k_values = range(1, 31)

for k in k_values:
    model_temp = KNeighborsClassifier(n_neighbors=k)
    model_temp.fit(X_train, y_train)
    predictions_temp = model_temp.predict(X_test)
    error = 1 - accuracy_score(y_test, predictions_temp)
    accuracy = accuracy_score(y_test, predictions_temp)
    error_rates.append(error)
    accuracies.append(accuracy)

optimal_k = list(k_values)[np.argmin(error_rates)]
min_error = min(error_rates)

print_success(f"Optimal K value found: {optimal_k}")
print_info(f"Lowest error rate: {min_error:.4f} ({(1-min_error)*100:.2f}% accuracy)")

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(k_values, error_rates, marker='o', linestyle='-', linewidth=2, color='red')
ax1.axvline(x=optimal_k, color='green', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
ax1.set_xlabel('K Value', fontsize=12)
ax1.set_ylabel('Error Rate', fontsize=12)
ax1.set_title('Elbow Method: Finding Optimal K', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(k_values, accuracies, marker='s', linestyle='-', linewidth=2, color='blue')
ax2.axvline(x=optimal_k, color='green', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
ax2.set_xlabel('K Value', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Accuracy vs K Value', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('step4_elbow_method.png', dpi=300, bbox_inches='tight')
print_success("Graph saved: step4_elbow_method.png")
plt.show()

# ================================================================
# STEP 5: TRAIN THE MODEL
# ================================================================
print_section("STEP 5: TRAIN THE MODEL")

print_info("Instantiating KNeighborsClassifier...")
model = KNeighborsClassifier(n_neighbors=optimal_k)
print_success("Model created")

print_info("Fitting model to training data...")
model.fit(X_train, y_train)
print_success("Model trained on 120 samples")

print_info("Making predictions on test data...")
y_pred = model.predict(X_test)
print_success("Predictions generated for 30 samples")

# ================================================================
# STEP 6: MODEL EVALUATION
# ================================================================
print_section("STEP 6: MODEL EVALUATION")

# A. Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{Colors.CYAN}Accuracy Score: {accuracy:.4f} ({accuracy*100:.2f}%){Colors.END}")

# B. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n{Colors.CYAN}Confusion Matrix:{Colors.END}")
print(cm)

# C. F1 Score
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_per_class = f1_score(y_test, y_pred, average=None)
print(f"\n{Colors.CYAN}F1 Score (Weighted): {f1_weighted:.4f}{Colors.END}")
print(f"F1 Scores per class: {f1_per_class}")

# D. Classification Report
print(f"\n{Colors.CYAN}Classification Report:{Colors.END}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ================================================================
# STEP 7: VISUALIZATIONS
# ================================================================
print_section("STEP 7: VISUALIZATIONS")

# Confusion Matrix Heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            cbar_kws={'label': 'Count'},
            ax=axes[0])
axes[0].set_title('Confusion Matrix - KNN Classifier', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Accuracy per class
classes = iris.target_names
accuracies_per_class = cm.diagonal() / cm.sum(axis=1)
colors_bar = ['#2ecc71', '#3498db', '#e74c3c']
axes[1].bar(classes, accuracies_per_class, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
axes[1].set_ylim([0, 1.1])
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
axes[1].axhline(y=accuracy, color='red', linestyle='--', linewidth=2, label='Overall Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (cls, acc) in enumerate(zip(classes, accuracies_per_class)):
    axes[1].text(i, acc + 0.05, f'{acc:.2%}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('step6_evaluation.png', dpi=300, bbox_inches='tight')
print_success("Graph saved: step6_evaluation.png")
plt.show()

# ================================================================
# STEP 8: CROSS-VALIDATION (BONUS)
# ================================================================
print_section("STEP 8: CROSS-VALIDATION (Bonus)")

cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print_info("5-Fold Cross-Validation Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")
print(f"\n{Colors.BLUE}Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f}){Colors.END}")

# ================================================================
# STEP 9: PREDICTION ON NEW DATA (EXAMPLE)
# ================================================================
print_section("STEP 9: MAKE PREDICTIONS ON NEW DATA")

# Create sample new iris flower
sample_iris = np.array([[5.1, 3.5, 1.4, 0.2]])  # Looks like Setosa
sample_iris_scaled = scaler.transform(sample_iris)
prediction = model.predict(sample_iris_scaled)
prediction_proba = model.predict(sample_iris_scaled)

print_info("Sample new iris flower measurements:")
print(f"  Sepal Length: 5.1 cm")
print(f"  Sepal Width: 3.5 cm")
print(f"  Petal Length: 1.4 cm")
print(f"  Petal Width: 0.2 cm")
print(f"\n{Colors.GREEN}Prediction: {iris.target_names[prediction[0]]}{Colors.END}")

# ================================================================
# FINAL SUMMARY
# ================================================================
print_section("PROJECT 2 COMPLETION SUMMARY")

print(f"""
{Colors.GREEN}✅ PROJECT SUCCESSFULLY COMPLETED!{Colors.END}

📊 KEY METRICS:
   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
   • F1 Score: {f1_weighted:.4f}
   • Optimal K: {optimal_k}
   • CV Mean Score: {cv_scores.mean():.4f}

📈 WHAT WE LEARNED:
   1. ✅ Loaded and understood the Iris dataset
   2. ✅ Scaled features for distance-based algorithm
   3. ✅ Split data into training and testing sets
   4. ✅ Found optimal K value using elbow method
   5. ✅ Trained KNN classifier
   6. ✅ Evaluated using confusion matrix and F1 score
   7. ✅ Validated using cross-validation

🚀 NEXT STEPS:
   1. Experiment with different algorithms (Decision Tree, SVM, etc.)
   2. Try different train-test splits (70-30, 90-10)
   3. Feature selection and engineering
   4. Hyperparameter tuning
   5. Deploy the model
   6. Build your portfolio on GitHub

📁 OUTPUT FILES:
   • step4_elbow_method.png
   • step6_evaluation.png

{Colors.BLUE}Thank you for completing Project 2!{Colors.END}
{Colors.CYAN}Your AI journey continues... 🚀{Colors.END}
""")

# ================================================================
# INTERPRETATION GUIDE
# ================================================================
print_section("HOW TO INTERPRET RESULTS")

print("""
🔍 CONFUSION MATRIX INTERPRETATION:
   - TP (True Positive): Diagonal values ✓ Correct predictions
   - FP (False Positive): Above diagonal ✗ Incorrectly predicted positive
   - FN (False Negative): Below diagonal ✗ Incorrectly predicted negative
   - TN (True Negative): Rest of diagonal ✓ Correct negative predictions

📊 ACCURACY MEANING:
   - 95%+ = Excellent
   - 90-95% = Very Good
   - 80-90% = Good
   - 70-80% = Acceptable
   - <70% = Needs improvement

🎯 F1 SCORE MEANING:
   - Balance between Precision (False Positives) and Recall (False Negatives)
   - Higher is better (range: 0 to 1)
   - More useful than accuracy for imbalanced datasets

⚙️ K-NEIGHBORS MEANING:
   - K=1: Sensitive to noise (overfitting)
   - K=3-5: Usually optimal for small datasets
   - K=100+: Too generic (underfitting)
   - Rule of thumb: K = √(number of samples)
""")

print(f"\n{Colors.GREEN}🎓 Project 2 completed! You're on your way to becoming an AI Engineer!{Colors.END}\n")
