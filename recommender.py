from flask import Flask, request, render_template
import pickle
import pandas as pd
from surprise import Dataset

app = Flask(__name__)

# Load the saved model from the pkl file
filename = 'model.pkl'
algorithm = pickle.load(open(filename, 'rb'))

# Load the data into a Pandas dataframe
data = pd.read_csv('ratings.csv')

# Load the movie titles into a Pandas dataframe
movies_df = pd.read_csv('movies.csv')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the user ID from the form
        user_id = int(request.form['user_id'])

        # Get the list of all movie IDs
        movies = data['movie_id'].unique()

        # Get the list of movie IDs already rated by the user
        user_movies = data[data['user_id'] == user_id]['movie_id']

        # Get the list of movie IDs not rated by the user
        new_movies = list(set(movies) - set(user_movies))

        # Create the test set
        testset = [[user_id, movie_id, 4.] for movie_id in new_movies]

        # Use the trained algorithm to generate movie recommendations for the user
        predictions = algorithm.test(testset)
        recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)

        # Get the movie titles for the recommended movies
        movie_titles = []
        for recommendation in recommendations[:10]:
            movie_titles.append(movies_df[movies_df['movieId'] == int(recommendation.iid)]['title'].values[0])

        # Return the movie recommendations to the user
        return render_template('index.html', recommendations=movie_titles)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
