import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample dataset
data = {
    'MovieID': [1, 2, 3, 4, 5],
    'Title': [
        'The Matrix', 
        'The Godfather', 
        'Pulp Fiction', 
        'The Dark Knight', 
        'Forrest Gump'
    ],
    'Genre': [
        'Action Sci-Fi', 
        'Crime Drama', 
        'Crime Drama', 
        'Action Crime', 
        'Drama Romance'
    ]
}

# Convert data to a DataFrame
movies = pd.DataFrame(data)

# Create a CountVectorizer to encode the genre information
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies['Genre'])

# Compute cosine similarity based on the genre matrix
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Function to get movie recommendations
def recommend_movies(title, cosine_sim=cosine_sim, movies=movies):
    if title not in movies['Title'].values:
        return "Movie not found in the dataset."
    
    # Get the index of the movie that matches the title
    idx = movies.index[movies['Title'] == title].tolist()[0]

    # Get similarity scores for all movies with the given movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar movies (excluding the input movie)
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movies
    return movies.iloc[movie_indices]['Title'].tolist()

# Example usage
user_input = 'The Matrix'
recommendations = recommend_movies(user_input)
print(f"Movies recommended for '{user_input}': {recommendations}")
