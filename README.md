# MovieMatch

MovieMatch is a small recommender systems project built around MovieLens 100K. A recruiter can search movies, rate a short list of titles, and get personalized recommendations in a Streamlit app. The offline training step learns movie embeddings from historical positive interactions, and the app infers a brand-new user profile from the movies rated during the session.

Live demo: [movie-match-app.streamlit.app](https://movie-match-app.streamlit.app/)

![MovieMatch screenshot](assets/streamlit-screenshot.png)

## What It Does

- Search movies by title substring
- Rate movies and manage a personal rating list in-session
- Get personalized recommendations from learned movie embeddings
- View similar movies for a selected title
- See simple recommendation explanations: “Because you liked …”

## Tech Stack

- Python 3.11+
- Streamlit
- NumPy, Pandas, SciPy
- `implicit` (ALS)

## Results Snapshot

Compared with a simple popularity recommender, the personalized ALS model:
- improved top-10 recommendation hit rate by about `2.2x` (`0.2463` vs `0.1125`)
- ranked good suggestions higher by about `2.1x` (`0.1253` vs `0.0605`)
- increased catalog coverage by about `9.6x` (`0.3704` vs `0.0386`)

## How It Works

- Historical MovieLens ratings are converted to implicit feedback (`rating >= 4`).
- ALS learns dense movie and user factors offline.
- For a new user, the app infers a user vector from rated movies via regularized least squares.
- Recommendations come from scoring unseen movies with dot products against the inferred user vector.
- Explanations use nearest rated movies in embedding space.

## Notes

- The MovieLens dataset is not committed to the repository.
- The trained model file `artifacts/model_artifacts.pkl` can be committed for Streamlit deployment.
- Downloaded dataset files under `artifacts/data/` should stay out of git.
- The recommendation explanations are intentionally simple and honest: they point to rated movies whose learned vectors are closest to the recommended movie.

## Attribution

This project uses the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset from GroupLens Research at the University of Minnesota. The dataset is downloaded locally by script and is not included in this repository.
