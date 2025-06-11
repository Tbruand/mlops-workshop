import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(path):
    df = pd.read_csv(path, sep=";")

    # 🔧 Suppression des colonnes non pertinentes
    df = df.drop(columns=['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'])

    # 🧼 Gestion des valeurs manquantes
    df = df.dropna(subset=['embarked'])
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())

    # 🔢 Encodage des variables catégorielles
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

    # 🎯 Séparation features / target
    X = df.drop(columns=['survived'])
    y = df['survived']

    # 🧪 Optionnel : normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ✂️ Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test