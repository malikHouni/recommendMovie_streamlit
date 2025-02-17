import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import numpy as np

# 1. Load and Prepare the Data

movies = pd.read_csv('./ml-10m/ml-10M100K/movies.dat', sep='::', header=None, names=['movieId', 'title', 'genres'], encoding='latin-1')
# Clean genres (important for TF-IDF)
movies['genres'] = movies['genres'].str.replace('|', ' ')

# 2. Feature Extraction (ou Chargement)
tfidf_model_filename = 'tfidf_model.pkl'
movie_features_filename = 'genre_vectors.pkl'  # On sauvegarde les vecteurs directement
genre_vectors = {}
tfidf_vectorizer = None  # Initialisation

if os.path.exists(tfidf_model_filename) and os.path.exists(movie_features_filename):
    # Load the model and features if they exist
    print("Loading pre-trained model and movie features...")
    with open(tfidf_model_filename, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open(movie_features_filename, 'rb') as f:
        genre_vectors = pickle.load(f)
else:
    st.write("Veuillez entraîner et enregistrer le modèle et les features en dehors de Streamlit.")
    st.stop() # On arrête l'exécution de l'app si le modèle n'est pas chargé

# 3. Streamlit UI

st.title("Recommandation de Films Basée sur vos Préférences")

# Sélection de 6 films
st.header("Choisissez 6 Films que vous Aimez")

# Affichage des titres de films pour la sélection
selected_movie_titles = st.multiselect(
    "Sélectionnez vos films préférés (6 maximum)",
    movies['title'].tolist(),
    max_selections=6
)

# Fonction pour créer le profil utilisateur
def create_user_profile(selected_movies, movies, genre_vectors, tfidf_vectorizer):
    """Crée un profil utilisateur en fonction des films sélectionnés."""
    if not selected_movies:
        return None

    movie_ids = movies[movies['title'].isin(selected_movies)]['movieId'].tolist()
    if not movie_ids:
        return None # No valid movie IDs

    # Créer un profil utilisateur basé sur la moyenne des features des films sélectionnés
    user_profile = np.zeros(len(genre_vectors[movie_ids[0]])) # Initialiser avec la dimension correcte
    for movie_id in movie_ids:
      user_profile += genre_vectors.get(movie_id, np.zeros(len(genre_vectors[movie_ids[0]]))) # On récupère le vecteur TF-IDF du film, si il est présent
    user_profile /= len(movie_ids)  # Moyenne
    return user_profile

# Fonction de recommandation
def recommend_movies(user_profile, genre_vectors, movies, top_n=10):
    """Recommande des films en fonction du profil utilisateur."""
    if user_profile is None:
        return "Veuillez sélectionner des films."

    movie_similarities = {}
    for movie_id, movie_features in genre_vectors.items():
        similarity = cosine_similarity([user_profile], [movie_features])[0][0]
        movie_similarities[movie_id] = similarity

    # Trier les films par similarité (décroissante)
    sorted_movies = sorted(movie_similarities.items(), key=lambda x: x[1], reverse=True)

    # Afficher les résultats
    recommendations = []
    for movie_id, similarity in sorted_movies[:top_n]:
        movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
        recommendations.append((movie_title, similarity))

    return recommendations

# Bouton pour générer les recommandations
if st.button("Obtenir les Recommandations"):
    # Créer le profil utilisateur
    user_profile = create_user_profile(selected_movie_titles, movies, genre_vectors, tfidf_vectorizer)

    # Générer les recommandations
    recommendations = recommend_movies(user_profile, genre_vectors, movies)

    # Afficher les résultats
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.header("Films Recommandés")
        for movie, similarity in recommendations:
            st.write(f"- {movie} (Similarité: {similarity:.2f})")