"""
Prediction models for the Online Course Enrollment Analysis project.

Reads course_enrollment_data.csv (created by course_enrollment_project.py) and
  1. predicts a student's final Status (Completed / In Progress / Dropped)
     with Logistic Regression, Decision Tree and Random Forest
  2. builds an early-warning drop-out model that uses only enrolment-time data

Install:  pip install pandas scikit-learn
Run:      python prediction_models.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, recall_score, roc_auc_score)

df = pd.read_csv("course_enrollment_data.csv")

# ---------------------------------------
# Part 1 : Predict final Status
# ---------------------------------------
number_cols = ["WatchedHours", "GenuineProgress", "CourseDurationHours"]
text_cols = ["Course", "Instructor", "Branch", "EnrollmentMonth"]

# Progress and ProgressGap are NOT used: they reveal the answer.
features = number_cols + text_cols
X = df[features]
y = df["Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print("Train rows:", len(X_train), "| Test rows:", len(X_test))
majority = y_test.value_counts(normalize=True).max() * 100
print(f"Majority guess accuracy: {majority:.1f}%\n")


def make_preprocess(scale_numbers):
    return ColumnTransformer([
        ("num", StandardScaler() if scale_numbers else "passthrough", number_cols),
        ("txt", OneHotEncoder(handle_unknown="ignore"), text_cols),
    ])


models = {
    "Logistic Regression": make_pipeline(make_preprocess(True),
                                         LogisticRegression(max_iter=1000)),
    "Decision Tree": make_pipeline(make_preprocess(False),
                                   DecisionTreeClassifier(max_depth=6, random_state=42)),
    "Random Forest": make_pipeline(make_preprocess(False),
                                   RandomForestClassifier(n_estimators=200, random_state=42,
                                                          n_jobs=-1)),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"{'Model':<22}{'Accuracy':>9}{'Precision':>11}{'Recall':>8}{'F1':>7}{'5-fold CV':>11}")
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted")
    cv_acc = cross_val_score(pipe, X, y, cv=cv).mean()
    print(f"{name:<22}{accuracy_score(y_test, pred)*100:>8.1f}%{prec*100:>10.1f}%"
          f"{rec*100:>7.1f}%{f1*100:>6.1f}%{cv_acc*100:>10.1f}%")

best = models["Decision Tree"]
labels = ["Completed", "In Progress", "Dropped"]
print("\nConfusion matrix - Decision Tree (rows = actual, columns = predicted)")
print(pd.DataFrame(confusion_matrix(y_test, best.predict(X_test), labels=labels),
                   index=labels, columns=labels))

# ---------------------------------------
# Part 2 : Early-warning drop-out model
# (uses only information known at enrolment)
# ---------------------------------------
early_text = ["Course", "Instructor", "Branch", "EnrollmentMonth"]
Xe = df[early_text + ["CourseDurationHours"]]
ye = (df["Status"] == "Dropped").astype(int)

Xe_train, Xe_test, ye_train, ye_test = train_test_split(
    Xe, ye, test_size=0.2, stratify=ye, random_state=42)

early = make_pipeline(
    ColumnTransformer([("txt", OneHotEncoder(handle_unknown="ignore"), early_text)],
                      remainder="passthrough"),
    LogisticRegression(max_iter=1000, class_weight="balanced"))
early.fit(Xe_train, ye_train)

pred_e = early.predict(Xe_test)
prob_e = early.predict_proba(Xe_test)[:, 1]
print("\nEARLY-WARNING DROP-OUT MODEL")
print(f"Drop-outs caught (recall): {recall_score(ye_test, pred_e)*100:.1f}%")
print(f"AUC score               : {roc_auc_score(ye_test, prob_e):.2f}")
print(f"Overall accuracy        : {(pred_e == ye_test).mean()*100:.1f}%")