import streamlit as st 
from data_preprocessing import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier 

# Charger et prétraiter les données
X_train, X_test, y_train, y_test, data = load_and_preprocess_data()

# Entraîner le modèle (Random Forest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Interface Streamlit
st.title("Prédiction de Maladie Cardiaque")
st.write("""
Cette application prédit la présence d'une maladie cardiaque en fonction des données saisies.
""")

# Formulaire de saisie
st.sidebar.header("Saisie des Données")
age = st.sidebar.slider("Âge", 0, 100, 50)
sex = st.sidebar.selectbox("Sexe", ["Femme", "Homme"])
cp = st.sidebar.selectbox("Type de Douleur Thoracique", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Pression Artérielle au Repos", 0, 200, 120)
chol = st.sidebar.slider("Cholestérol Sérique", 0, 600, 200)
fbs = st.sidebar.selectbox("Glycémie à Jeun > 120 mg/dl", [0, 1])
restecg = st.sidebar.selectbox("Résultats Électrocardiographiques au Repos", [0, 1, 2])
thalach = st.sidebar.slider("Fréquence Cardiaque Maximale Atteinte", 0, 220, 150)
exang = st.sidebar.selectbox("Angine Induite par l'Exercice", [0, 1])
oldpeak = st.sidebar.slider("Dépression du Segment ST Induite par l'Exercice", 0.0, 6.2, 1.0)
slope = st.sidebar.selectbox("Pente du Segment ST au Pic de l'Exercice", [0, 1, 2])
ca = st.sidebar.slider("Nombre de Gros Vaisseaux Colorés par Fluoroscopie", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassémie", [3, 6, 7])

# Prédiction
if st.sidebar.button("Prédire"):
    input_data = [[
        age, 1 if sex == "Homme" else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    ]]
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("Résultat de la Prédiction")
    if prediction[0] == 1:
        st.write("**Maladie Cardiaque détectée.**")
    else:
        st.write("**Pas de maladie cardiaque détectée.**")

    st.subheader("Probabilités")
    st.write(f"Probabilité de maladie cardiaque : {prediction_proba[0][1]:.2f}")
    st.write(f"Probabilité de non-maladie cardiaque : {prediction_proba[0][0]:.2f}")