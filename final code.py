import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from surprise.model_selection import KFold, train_test_split
from surprise.accuracy import rmse, mae
from surprise import Reader, Dataset, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import fuzz

# Load data
merged_data = pd.read_csv('merged_data.csv')
movies = pd.read_csv('dataset.csv')

# Data preprocessing
movies['release_date'] = pd.to_datetime(movies['release_date'])
movies['genres'] = movies['genre'].str.split(',')
movies = movies.rename(columns={'id': 'MovieID'})
features = ['title', 'genres', 'original_language', 'popularity', 'release_date', 'vote_average', 'vote_count']
df_model = movies[features]
df_model['overview'] = movies['overview'].fillna('')
df_model['genre'] = movies['genre'].fillna('')

# Exploratory Data Analysis (EDA)
# Distribution of Movie Genres
plt.figure(figsize=(12, 6))
genres_count = movies['genre'].str.split(',').explode().value_counts()
sns.barplot(x=genres_count.values, y=genres_count.index, palette='viridis')
plt.title('Distribution of Movie Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# Distribution of Movie Release Years
plt.figure(figsize=(10, 6))
movies['release_year'] = movies['release_date'].dt.year
sns.histplot(movies['release_year'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Movie Release Years')
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.show()

# Distribution of Movie Ratings
plt.figure(figsize=(10, 6))
sns.histplot(movies['vote_average'], bins=20, kde=True, color='salmon')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Top Genres by Popularity
top_genres = movies.groupby('genre')['popularity'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='mako')
plt.title('Top Genres by Popularity')
plt.xlabel('Average Popularity')
plt.ylabel('Genre')
plt.show()

# Release Date Trends
release_year_count = movies['release_year'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
release_year_count.plot(kind='line', marker='o', color='orange')
plt.title('Release Date Trends')
plt.xlabel('Release Year')
plt.ylabel('Number of Movies Released')
plt.grid(True)
plt.show()

# Vote Count vs. Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='vote_count', y='vote_average', data=movies, color='green', alpha=0.5)
plt.title('Vote Count vs. Rating')
plt.xlabel('Vote Count')
plt.ylabel('Average Rating')
plt.show()

# Language Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='original_language', data=movies, palette='rocket')
plt.title('Language Distribution')
plt.xlabel('Original Language')
plt.ylabel('Count')
plt.show()

# Rating Distribution Over Time
rating_over_time = movies.groupby('release_year')['vote_average'].mean()
plt.figure(figsize=(12, 6))
rating_over_time.plot(kind='line', marker='o', color='purple')
plt.title('Rating Distribution Over Time')
plt.xlabel('Release Year')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()

# Collaborative Filtering
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(merged_data[['CustomerID', 'MovieID', 'Rating']], reader)
algo = SVD()
kf = KFold(n_splits=5)
rmse_list = []
mae_list = []
collaborative_recs = {}

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    testset_user_ids = set([uid for uid, _, _ in testset])
    for user_id in testset_user_ids:
        user_recs = algo.test([(user_id, iid, None) for iid in merged_data['MovieID'].unique()])
        collaborative_recs[user_id] = [(int(pred.iid), pred.est) for pred in user_recs]
    trainset, testset = train_test_split(data, test_size=0.2)
    predictions = algo.test(testset)
    rmse_list.append(rmse(predictions))
    mae_list.append(mae(predictions))

avg_rmse = sum(rmse_list) / len(rmse_list)
avg_mae = sum(mae_list) / len(mae_list)

print("Average RMSE:", avg_rmse)
print("Average MAE:", avg_mae)

# Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_model['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim, df=df_model):
    idx = df.index[df['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

def get_hybrid_recommendations(input_value):
    if input_value.isdigit():
        CustomerID = int(input_value)
        if CustomerID in collaborative_recs:
            print("Collaborative Filtering Recommendations for CustomerID", CustomerID)
            for i, (movie_id, est_rating) in enumerate(collaborative_recs[CustomerID][:10]):
                print(f"Recommendation {i+1}: Movie ID {movie_id}, Estimated Rating {est_rating}")
        else:
            print(f"No collaborative filtering recommendations found for CustomerID {CustomerID}")
    else:
        if input_value.isdigit():
            movie_id = int(input_value)
            movie_title = get_movie_title_from_id(movie_id)
        else:
            movie_title = input_value

        recommendations = get_recommendations(movie_title)
        print("\nContent-Based Filtering Recommendations for '{}':".format(movie_title))
        print(recommendations)

# Example usage:
input_value = input("Enter a Movie Title, Movie ID, or CustomerID: ")
get_hybrid_recommendations(input_value)