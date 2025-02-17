import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os # Import pour vérifier si le fichier existe
import numpy as np # Import pour gérer les vecteurs

# 1. Load and Prepare the Data

movies = pd.read_csv('./ml-10m/ml-10M100K/movies.dat', sep='::', header=None, names=['movieId', 'title', 'genres'], encoding='latin-1')

# Clean genres (important for TF-IDF)
movies['genres'] = movies['genres'].str.replace('|', ' ')

# 2. Feature Extraction et entraînement du modèle TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

# 3. Création des vecteurs de genres (genre_vectors)
genre_vectors = {}
for index, row in movies.iterrows():
    movie_id = row['movieId']
    movie_index = movies[movies['movieId'] == movie_id].index[0] # Récupérer l'index du film
    genre_vectors[movie_id] = tfidf_matrix[movie_index].toarray().flatten()  # Convertir en array et 'flatten'

# 4. Enregistrement du modèle et des vecteurs
model_filename = 'tfidf_model.pkl'
movie_features_filename = 'genre_vectors.pkl'

# Enregistrement du modèle
with open(model_filename, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Enregistrement des vecteurs de genres
with open(movie_features_filename, 'wb') as f:
    pickle.dump(genre_vectors, f)

print(f"Modèle TF-IDF et features enregistrés avec succès dans {model_filename} et {movie_features_filename}")
print("Exécuter maintenant l'application Streamlit.")