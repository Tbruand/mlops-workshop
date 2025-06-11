from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from joblib import dump
from preprocess import preprocess
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Préparation des données
X_train, X_test, y_train, y_test = preprocess("data/titanic.csv")

# Entraînement du modèle
model = RandomForestClassifier(n_estimators=5, random_state=42)
model.fit(X_train, y_train)

# Prédictions
preds = model.predict(X_test)
probas = model.predict_proba(X_test)[:, 1]  # pour courbe ROC

# Calcul des métriques
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
auc = roc_auc_score(y_test, probas)

# Affichage console
print(f"Accuracy : {acc:.2f}")
print(f"Precision : {prec:.2f}")
print(f"Recall : {rec:.2f}")
print(f"F1-score : {f1:.2f}")
print(f"AUC : {auc:.2f}")
print("\nClassification report :\n", classification_report(y_test, preds))

# Sauvegarde des métriques
os.makedirs("models/metrics", exist_ok=True)
with open("models/metrics/score.txt", "w") as f:
    f.write(f"Accuracy : {acc:.2f}\n")
    f.write(f"Precision : {prec:.2f}\n")
    f.write(f"Recall : {rec:.2f}\n")
    f.write(f"F1-score : {f1:.2f}\n")
    f.write(f"AUC : {auc:.2f}\n")

# Sauvegarde du modèle
dump(model, "models/model.joblib")

# Matrice de confusion
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.savefig("models/metrics/confusion_matrix.png")
plt.close()

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, probas)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Courbe ROC")
plt.xlabel("Faux positifs")
plt.ylabel("Vrais positifs")
plt.legend()
plt.tight_layout()
plt.savefig("models/metrics/roc_curve.png")
plt.close()