# 🧪 MLOps Workshop

Ce projet est un exemple pédagogique de pipeline MLOps minimal, intégrant :

* Le prétraitement des données,
* L'entraînement d'un modèle de machine learning,
* La sauvegarde des résultats (score + modèle),
* Un workflow CI avec GitHub Actions.

---

## 📁 Structure du projet

```
mlops-workshop/
├── data/                 # Contient le jeu de données (ex: iris.csv)
├── models/               # Contient le modèle entraîné et les métriques
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

* Lit `data/iris.csv`
* Prétraite les données (`train_test_split`)
* Entraîne un `RandomForestClassifier`
* Sauvegarde le score dans `models/score.txt`
* Sauvegarde le modèle dans `models/model.joblib`

---

## 🔁 Intégration Continue (CI)

Un workflow GitHub Actions est défini dans `.github/workflows/ci.yml` pour :

* Installer les dépendances
* Lancer l'entraînement
* Uploader les résultats d'accuracy en tant qu’artifact

---

## 📦 Dépendances

* `pandas`
* `scikit-learn`
* `joblib`

---

## ✍️ Auteur

Thomas Bruand – Projet de formation MLOps
