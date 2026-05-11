import pickle, numpy as np, pandas as pd, os
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

csv_path = os.path.join(os.path.dirname(os.getcwd()), "data_csv", "raw", "vitamin_deficiency_disease_dataset_20260123.ods")
df = pd.read_excel(csv_path, engine="odf")
cols_to_drop = [c for c in ["age","gender","bmi","smoking_status","alcohol_consumption","exercise_level","diet_type","sun_exposure","latitude_region","income_level","symptoms_list","symptoms_count"] if c in df.columns]
df = df.drop(columns=cols_to_drop)
le = LabelEncoder()
y = le.fit_transform(df["disease_diagnosis"])
X = df.drop(columns=["disease_diagnosis"])
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

with open("models_v3/best_model_v3.pkl", "rb") as f:
    model = pickle.load(f)

y_score = model.predict_proba(X_test)
y_pred = model.predict(X_test)
y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))

print("=== AUC par classe (RF v3) ===")
for i, cls in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    print("  %-25s AUC = %.4f" % (cls, auc(fpr, tpr)))

print("\n=== Matrice de confusion RF v3 ===")
print("Classes:", le.classes_.tolist())
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n=== Feature importances top 10 ===")
fi = model.named_steps["rf"].feature_importances_
top = sorted(zip(X.columns.tolist(), fi), key=lambda x: x[1], reverse=True)[:10]
for name, imp in top:
    print("  %-35s %.4f" % (name, imp))
