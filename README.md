# 🔪 MLOps Workshop

Ce projet est un exemple pédagogique de pipeline MLOps minimal, intégrant :

* Le prétraitement des données,
* L'entraînement d'un modèle de machine learning,
* La sauvegarde des résultats (score + modèle + visualisations),
* Un workflow CI avec GitHub Actions.

---

## 📁 Structure du projet

```
mlops-workshop/
├── data/                 # Contient le jeu de données (ex: titanic.csv)
├── models/               # Contient le modèle entraîné
│   └── metrics/          # Contient les métriques (score.txt, courbes ROC/confusion)
├── notebooks/            # Analyses exploratoires (EDA)
├── src/                  # Code source (prétraitement + entraînement)
│   ├── preprocess.py
│   └── train.py
├── .github/workflows/    # Workflows GitHub Actions
│   └── ci.yml
├── requirements.txt      # Dépendances Python
└── README.md             # Ce fichier
```

---

## ⚙️ Installation

```bash
# 1. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # ou .\venv\Scripts\activate sous Windows

# 2. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Lancer le pipeline

```bash
# Lancer l'entraînement du modèle
python src/train.py
```

Cela :

* Lit `data/titanic.csv`
* Prétraite les données (`train_test_split`, `StandardScaler`, encodage...)
* Entraîne un `RandomForestClassifier`
* Sauvegarde le modèle dans `models/model.joblib`
* Sauvegarde les métriques dans `models/metrics/score.txt`
* Génère deux visualisations :

  * `confusion_matrix.png`
  * `roc_curve.png`

---

## 🔁 Intégration Continue (CI)

Un workflow GitHub Actions est défini dans `.github/workflows/ci.yml` pour :

* Installer les dépendances
* Lancer l'entraînement
* Uploader le fichier `models/metrics/score.txt` en tant qu’artifact

---

## 📦 Dépendances

* `pandas`
* `scikit-learn`
* `joblib`
* `matplotlib`
* `seaborn`

---

## ✍️ Auteur

Thomas Bruand – Projet de formation MLOps
