import pandas as pd
import numpy as np

movies_columns = ["movie_id", "title", "genres"]
ratings_columns = ["user_id", "movie_id", "rating", "timestamp"]
users_columns = ["user_id", "gender", "age", "occupation", "zip_code"]

movies = pd.read_csv("./data/movies.dat", sep="::", engine="python", names=movies_columns, encoding="ISO-8859-1") 
# encoding="" present to deal with special characters that may be present in the dataset e.g accents, symbols and non-ASCII characters
# sep="::" because of the data's structure. Columns are separated by double columns, not commas
# engine="python" is needed because panda's default c engine does not support multicharacter delimiters e.g "::"
ratings = pd.read_csv("./data/ratings.dat", sep="::", engine="python", names=ratings_columns, encoding="ISO-8859-1")
users = pd.read_csv("./data/users.dat", sep="::", engine="python", names=users_columns, encoding="ISO-8859-1")

# Convert explicit ratings to implicit feedback
positive_interactions = ratings[ratings["rating"] >= 4][["user_id", "movie_id"]].copy()
# This code takes all the interactions where the rating is greater than or equal to 4 and stores it in a variable
positive_interactions["label"] = 1
# This code converts all the explicit positive ratings to a 1

# For this study we are using implicit feedback, which means that any ratings below a 4 are completely and utterly irrelevant 
# to this study. A rating less than 4 would be ambiguous and doesn't tell us much. The negative samples will come from movies
# that the user has watched but did not bother to review. We assume movies a user never rated are the ones they disliked to
# prevent bias.

all_unique_users = ratings["user_id"].unique()
all_unique_movies = movies["movie_id"].unique()
# Here we get all the unique users and unique movies

user_movie_interactions = ratings.groupby("user_id")["movie_id"].apply(set).to_dict()
# Here we create a dictionary (basically a hash map) mapping unique user_id's to a set of unique movies that they have 
# explicity rated


# Function to sample negative interactions safely
def sample_negative_movies(user_id, num_samples):
    interacted_movies = user_movie_interactions.get(user_id, set())
    non_interacted_movies = list(set(all_unique_movies) - interacted_movies)
    
    # Ensure we do not sample more than available movies
    num_samples = min(len(non_interacted_movies), num_samples)

    return np.random.choice(non_interacted_movies, size=num_samples, replace=False) if num_samples > 0 else []


# Generate negative samples for each user
negative_samples = []
for user in all_unique_users:
    num_positive = len(user_movie_interactions.get(user, []))
    if num_positive > 0:
        sampled_movies = sample_negative_movies(user, num_positive)
        for movie in sampled_movies:
            negative_samples.append((user, movie, 0))

# Convert negative samples into a DataFrame
negative_interactions = pd.DataFrame(negative_samples, columns=["user_id", "movie_id", "label"])

# Step 3: Combine positive and negative interactions
implicit_feedback = pd.concat([positive_interactions, negative_interactions], ignore_index=True)


print(implicit_feedback.head())

print("Implicit feedback dataset created successfully!")