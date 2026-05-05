# Anime-Recommender-System
The smart website anime recommendations tailored by genre, rating, and popularity.

## ALS Recommender

File `als_recommender.py` implements an explicit ALS model using `anime data/rating.csv` and enriches the output with `anime data/anime.csv`.

Run a quick training test:

```bash
python als_recommender.py --max-rows 5000 --n-factors 16 --n-iters 2 --user-id 1 --top-k 5
```

The trained model is saved under `artifacts/als_model/als_model.pkl`.

Find similar anime by id:

```bash
python als_recommender.py --max-rows 5000 --n-factors 16 --n-iters 2 --anime-id 5114 --top-k 5
```

Find similar anime by name:

```bash
python als_recommender.py --max-rows 5000 --n-factors 16 --n-iters 2 --anime-name "Fullmetal Alchemist: Brotherhood" --top-k 5
```

## Streamlit App

Launch the web app from the project root:

```bash
streamlit run app.py
```

In the app, you can choose an anime by name or enter an `anime_id`, then it will return the 5 most similar anime.
