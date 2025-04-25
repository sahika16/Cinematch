from flask import Flask, render_template, request
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  

app = Flask(__name__)

def load_data():
    movies = pd.read_csv('movies.csv')
    credits = pd.read_csv('credits.csv')
    return movies.merge(credits, on='title')

def preprocess_data(df):
    df = df[['movie_id', 'title', 'genres', 'keywords', 'cast', 'crew', 'overview']]
    df.dropna(inplace=True)

    def process_text(text):
        return [i['name'] for i in ast.literal_eval(text)]

    df['genres'] = df['genres'].apply(process_text)
    df['keywords'] = df['keywords'].apply(process_text)
    df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3])  # Limit to top 3 cast
    df['crew'] = df['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])
    df['overview'] = df['overview'].apply(lambda x: x.split())

    df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
    df['tags'] = df['tags'].apply(lambda x: ' '.join(x).lower())

    return df

movies = load_data()
processed_movies = preprocess_data(movies.copy())

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(processed_movies['tags'])
similarity = cosine_similarity(tfidf_matrix)

def recommend(movie_title, api_key="a074373ef7f3e929489db7b98c014185"):
    try:
        all_titles = processed_movies['title'].tolist()
        
        match = process.extractOne(movie_title, all_titles)  
        if not match or match[1] < 50:  
            return None, "Movie not found. Please check the spelling or try another title.", None

        best_match = match[0]
        movie_index = processed_movies[processed_movies['title'] == best_match].index[0]
        sim_scores = list(enumerate(similarity[movie_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 recommendations

        recommendations = []
        for i in sim_scores:
            movie_id = processed_movies.iloc[i[0]]['movie_id']
            title = processed_movies.iloc[i[0]]['title']
            poster_url = None
            try:
                response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}")
                poster_path = response.json().get('poster_path')
                poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else None
            except:
                pass

            recommendations.append({
                'title': title,
                'poster': poster_url or "https://via.placeholder.com/500x750?text=No+Poster+Available"
            })

        return recommendations, None, best_match

    except Exception as e:
        return None, f"Error: {str(e)}", None

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error = ""
    corrected_name = ""
    if request.method == "POST":
        movie = request.form["movie"]
        if movie:
            results, error, corrected_name = recommend(movie)

    return render_template("index.html", results=results, error=error, corrected_name=corrected_name)

if __name__ == "__main__":
    app.run(debug=True)
