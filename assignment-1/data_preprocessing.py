import pandas as pd

movies_columns = ["movie_id", "title", "genres"]
ratings_columns = ["user_id", "movie_id", "rating", "timestamp"]
users_columns = ["user_id", "gender", "age", "occupation", "zip_code"]

movies = pd.read_csv("./data/movies.dat", sep="::", engine="python", names=movies_columns, encoding="ISO-8859-1") 
# encoding="" present to deal with special characters that may be present in the dataset e.g accents, symbols and non-ASCII characters
# sep="::" because of the data's structure. Columns are separated by double columns, not commas
# engine="python" is needed because panda's default c engine does not support multicharacter delimiters e.g "::"
ratings = pd.read_csv("./data/ratings.dat", sep="::", engine="python", names=ratings_columns, encoding="ISO-8859-1")
users = pd.read_csv("./data/users.dat", sep="::", engine="python", names=users_columns, encoding="ISO-8859-1")

print("Movies DataFrame:")
print(movies.head())

print("\nRatings DataFrame:")
print(ratings.head())

print("\nUsers DataFrame:")
print(users.head())