import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   OneHotEncoder, OrdinalEncoder, LabelEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import joblib

# Configuration de la page
st.set_page_config(page_title="Data Preprocessing Pro", layout="wide")
sns.set(style="whitegrid")

# Fonctions de prétraitement
def handle_missing_data(df):
    st.subheader("Gestion des valeurs manquantes")
    
    methods = st.multiselect("Méthodes de traitement des NaN", [
        'Supprimer les lignes', 
        'Imputation moyenne/médiane/mode', 
        'KNN Imputer'
    ])
    
    if 'Supprimer les lignes' in methods:
        df = df.dropna()
    
    if 'Imputation moyenne/médiane/mode' in methods:
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns
        
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                method = st.selectbox(f"Méthode pour {col}", ['mean', 'median', 'constant'])
                imp = SimpleImputer(strategy=method, fill_value=0 if method == 'constant' else None)
                df[col] = imp.fit_transform(df[[col]])
        
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    if 'KNN Imputer' in methods:
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            n_neighbors = st.number_input("Nombre de voisins KNN", 2, 10, 3)
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            df[num_cols] = knn_imputer.fit_transform(df[num_cols])
    
    return df

def encode_categorical(df):
    st.subheader("Encodage des variables catégorielles")
    cat_cols = df.select_dtypes(exclude=np.number).columns
    
    for col in cat_cols:
        method = st.selectbox(f"Encodage pour {col}", 
                            ['One-Hot', 'Ordinal', 'Label', 'Target'], 
                            key=f'enc_{col}')
        
        if method == 'One-Hot':
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
        
        elif method == 'Ordinal':
            categories = sorted(df[col].unique())
            ord_map = {k:i for i,k in enumerate(categories)}
            df[col] = df[col].map(ord_map)
        
        elif method == 'Label':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        
        elif method == 'Target':
            if 'target' in df.columns:
                means = df.groupby(col)['target'].mean()
                df[col] = df[col].map(means)
    
    return df

def scale_features(df):
    st.subheader("Normalisation des données")
    num_cols = df.select_dtypes(include=np.number).columns
    scaler_type = st.selectbox("Type de normalisation", [
        'StandardScaler', 
        'MinMaxScaler', 
        'RobustScaler'
    ])
    
    scaler = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }[scaler_type]
    
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def handle_outliers(df):
    st.subheader("Détection et traitement des outliers")
    num_cols = df.select_dtypes(include=np.number).columns
    method = st.selectbox("Méthode de détection", [
        'IQR', 
        'Z-Score', 
        'Isolation Forest', 
        'Local Outlier Factor'
    ])
    
    if method == 'IQR':
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    
    elif method == 'Z-Score':
        threshold = st.number_input("Seuil Z-Score", 2.0, 5.0, 3.0)
        for col in num_cols:
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z < threshold]
    
    elif method == 'Isolation Forest':
        iso = IsolationForest(contamination=0.05)
        preds = iso.fit_predict(df[num_cols])
        df = df[preds == 1]
    
    elif method == 'Local Outlier Factor':
        lof = LocalOutlierFactor()
        preds = lof.fit_predict(df[num_cols])
        df = df[preds == 1]
    
    return df

def feature_selection(df, target_col=None):
    st.subheader("Sélection de caractéristiques")
    num_cols = df.select_dtypes(include=np.number).columns
    
    method = st.selectbox("Méthode de sélection", [
        'Variance Threshold',
        'SelectKBest',
        'Corrélation'
    ])
    
    if method == 'Variance Threshold':
        threshold = st.slider("Seuil de variance", 0.0, 1.0, 0.01)
        selector = VarianceThreshold(threshold=threshold)
        selected = selector.fit_transform(df[num_cols])
        df = pd.DataFrame(selected, columns=num_cols[selector.get_support()])
    
    elif method == 'SelectKBest':
        k = st.number_input("Nombre de caractéristiques à garder", 1, len(num_cols), len(num_cols))
        selector = SelectKBest(mutual_info_classif, k=k)
        selected = selector.fit_transform(df[num_cols], df[target_col] if target_col else None)
        df = pd.DataFrame(selected, columns=num_cols[selector.get_support()])
    
    elif method == 'Corrélation':
        corr_threshold = st.slider("Seuil de corrélation", 0.0, 1.0, 0.8)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        df = df.drop(to_drop, axis=1)
    
    return df

# Interface utilisateur
st.title("🛠 Data Preprocessing Pro")
st.markdown("### Un outil complet pour préparer vos données en 1 clic")

# Upload de données
uploaded_file = st.file_uploader("Téléchargez votre dataset", type=["csv", "xlsx", "txt"])
df = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep='\t')
        
        st.success("Données chargées avec succès !")
        
        # Affichage des données brutes
        st.subheader("Aperçu des données")
        st.dataframe(df.head())
        
        # Sélection de la colonne cible
        target_col = st.selectbox("Sélectionnez la colonne cible (optionnel)", 
                                [''] + list(df.columns))
        
        # Sidebar pour les options de prétraitement
        st.sidebar.header("Options de prétraitement")
        steps = st.sidebar.multiselect("Étapes à appliquer", [
            'Gestion des valeurs manquantes',
            'Encodage des catégorielles',
            'Traitement des outliers', 
            'Normalisation',
            'Sélection de caractéristiques'
        ])
        
        # Application des étapes sélectionnées
        if 'Gestion des valeurs manquantes' in steps:
            df = handle_missing_data(df)
            
        if 'Encodage des catégorielles' in steps:
            df = encode_categorical(df)
            
        if 'Traitement des outliers' in steps:
            df = handle_outliers(df)
            
        if 'Normalisation' in steps:
            df = scale_features(df)
            
        if 'Sélection de caractéristiques' in steps:
            df = feature_selection(df, target_col if target_col else None)
        
        # Affichage des données transformées
        st.subheader("Données transformées")
        st.dataframe(df.head())
        
        # Téléchargement des données nettoyées
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger les données nettoyées",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv'
        )
        
        # Visualisations
        st.subheader("Analyse des données")
        
        if target_col and target_col in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(x=target_col, data=df, ax=ax)
            st.pyplot(fig)
        
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            col = st.selectbox("Sélectionnez une colonne pour l'histogramme", num_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Erreur : {str(e)}")
else:
    st.info("Veuillez télécharger un fichier CSV, Excel ou texte pour commencer")

