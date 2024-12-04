# Movie Recommendation System

This project implements a **Collaborative Filtering-based Recommendation System** using the MovieLens dataset. It preprocesses the data, applies normalization and feature engineering, and builds a recommender system to suggest movies based on user preferences.

---

### **Project Structure**

1. **Preprocessing**:
   - Import libraries and load datasets.
   - Split and encode genres using one-hot encoding.
   - Normalize ratings and convert timestamps to datetime.
   - Filter out movies with insufficient ratings.

2. **Model Building**:
   - Create user-item interaction matrices.
   - Apply collaborative filtering techniques (SVD).

3. **Evaluation**:
   - Compute RMSE for validation.
   - Recommend movies to users based on predicted ratings.

---

### **Prerequisites**

- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

### **How to Use**

1. **Run Preprocessing**:
   Load the dataset, clean and preprocess data by running the cells in the notebook.

2. **Train the Model**:
   Use matrix factorization techniques to train the collaborative filtering model.

3. **Generate Recommendations**:
   Provide a user ID and get a list of recommended movies.

4. **Evaluate Performance**:
   Use the evaluation cells to validate the model's accuracy.

---

### **Dataset**

The system uses the [**MovieLens 32M Dataset**](https://grouplens.org/datasets/movielens/32m/), which contains:
- Ratings: User ratings for movies.
- Movies: Metadata about movies, including genres.
- Tags: User-provided tags for movies.
- Links: Links to IMDb and TMDb webpages

---

### **Notebook Overview**

#### **Code Highlights**:
- **Preprocessing**:
  ```python
  from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
  movies['genres'] = movies['genres'].str.split('|')
  mlb = MultiLabelBinarizer()
  genres_encoded = mlb.fit_transform(movies['genres'])
  genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
  movies = movies.join(genres_df)
  ```
- **Normalization**:
  ```python
  scaler = MinMaxScaler()
  ratings['rating_normalized'] = scaler.fit_transform(ratings['rating'].values.reshape(-1, 1))
  ```
- **Filtering**:
  ```python
  min_movie_ratings = 5
  movie_counts = ratings['movieId'].value_counts()
  ratings = ratings[ratings['movieId'].isin(movie_counts[movie_counts >= min_movie_ratings].index)]
  ```

#### **Visualization**:
Visualize the rating distributions, genres, and movie popularity using `matplotlib` and `seaborn`.

---

### **Future Enhancements**

- **Hybrid Models**:
  Combine collaborative filtering with content-based techniques.
- **Neural Networks**:
  Implement neural collaborative filtering for improved performance.
- **Personalized Features**:
  Include temporal dynamics and user metadata.

---
