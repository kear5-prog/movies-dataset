import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="IMDb Audience Rating Predictor",
    layout="wide"
)

st.title("🎬 IMDb Audience Rating Prediction App")
st.markdown(
    """
    This application explores IMDb movie data and predicts **audience ratings**
    based on critic ratings, genre, and release year using a machine learning model.
    """
)

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/Movie Rating IMDB Dataset.csv")

df = load_data()

# --------------------------------------------------
# Basic data cleaning
# --------------------------------------------------
df = df.dropna(subset=["imdb_rating", "metascore", "genre", "year"])
df["year"] = df["year"].astype(int)

# --------------------------------------------------
# Sidebar: Data filters (EDA only)
# --------------------------------------------------
st.sidebar.header("📊 Data Filters (Exploration Only)")

sidebar_genres = st.sidebar.multiselect(
    "Select genre(s)",
    options=sorted(df["genre"].unique()),
    default=sorted(df["genre"].unique())
)

sidebar_years = st.sidebar.slider(
    "Select year range",
    int(df["year"].min()),
    int(df["year"].max()),
    (2000, int(df["year"].max()))
)

filtered_df = df[
    (df["genre"].isin(sidebar_genres)) &
    (df["year"].between(sidebar_years[0], sidebar_years[1]))
]

# --------------------------------------------------
# Exploratory Data Analysis
# --------------------------------------------------
st.subheader("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**IMDb Audience Rating Distribution**")
    fig, ax = plt.subplots()
    ax.hist(filtered_df["imdb_rating"], bins=20)
    ax.set_xlabel("IMDb Rating")
    ax.set_ylabel("Number of Movies")
    st.pyplot(fig)

with col2:
    st.markdown("**Audience vs Critic Ratings**")
    fig, ax = plt.subplots()
    ax.scatter(
        filtered_df["metascore"],
        filtered_df["imdb_rating"],
        alpha=0.5
    )
    ax.set_xlabel("Metascore (Critics)")
    ax.set_ylabel("IMDb Rating (Audience)")
    st.pyplot(fig)

# --------------------------------------------------
# Machine Learning Model
# --------------------------------------------------
st.subheader("🤖 Audience Rating Prediction")
st.markdown(
    "Use the controls below to predict the **IMDb audience rating**. "
    "Predictions are based on patterns learned from the full dataset."
)

features = ["metascore", "genre", "year"]
target = "imdb_rating"

X = df[features]
y = df[target]

preprocessor = ColumnTransformer(
    transformers=[
        ("genre", OneHotEncoder(handle_unknown="ignore"), ["genre"]),
        ("num", "passthrough", ["metascore", "year"])
    ]
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

st.info(f"📉 Model Mean Absolute Error (MAE): {mae:.2f}")

# --------------------------------------------------
# Prediction Inputs
# --------------------------------------------------
st.markdown("### 🎯 Make a Prediction")

col1, col2, col3 = st.columns(3)

# Genre input
genre_options = ["All genres"] + sorted(df["genre"].unique())
with col1:
    user_genre = st.selectbox("Genre", genre_options)

# Critic score input
with col2:
    user_metascore = st.slider(
        "Critic Rating (Metascore)",
        0, 100, 70
    )

# Year input
year_options = ["All years"] + sorted(df["year"].unique())
with col3:
    user_year = st.selectbox("Release Year", year_options)

# --------------------------------------------------
# Handle "All" selections for model input
# --------------------------------------------------
if user_genre == "All genres":
    model_genre = df["genre"].mode()[0]
else:
    model_genre = user_genre

if user_year == "All years":
    model_year = int(df["year"].median())
else:
    model_year = int(user_year)

user_input = pd.DataFrame(
    {
        "metascore": [user_metascore],
        "genre": [model_genre],
        "year": [model_year]
    }
)

prediction = pipeline.predict(user_input)[0]

# --------------------------------------------------
# Display prediction
# --------------------------------------------------
st.success(f"⭐ Predicted IMDb Audience Rating: **{prediction:.1f} / 10**")

# --------------------------------------------------
# Interpretation text (dynamic & consistent)
# --------------------------------------------------
if user_genre == "All genres" and user_year == "All years":
    interpretation = (
        f"Based on patterns learned from thousands of movies across **all genres** "
        f"and **all release years**, films with a critic rating near "
        f"**{user_metascore}** typically receive an audience rating of around "
        f"**{prediction:.1f}**."
    )

elif user_genre == "All genres":
    interpretation = (
        f"Based on patterns learned from thousands of movies across **all genres**, "
        f"films released around **{user_year}** with a critic rating near "
        f"**{user_metascore}** typically receive an audience rating of around "
        f"**{prediction:.1f}**."
    )

elif user_year == "All years":
    interpretation = (
        f"Based on patterns learned from thousands of **{user_genre}** movies across "
        f"**all release years**, films with a critic rating near "
        f"**{user_metascore}** typically receive an audience rating of around "
        f"**{prediction:.1f}**."
    )

else:
    interpretation = (
        f"Based on patterns learned from thousands of past movies, "
        f"**{user_genre}** movies released around **{user_year}** with a critic "
        f"rating near **{user_metascore}** typically receive an audience rating "
        f"of around **{prediction:.1f}**."
    )

st.markdown(f"**Interpretation:**  \n{interpretation}")

st.caption(
    "Note: Data filters affect visualizations only. Predictions are statistical estimates "
    "based on historical IMDb data and may differ from actual audience ratings."
)

# --------------------------------------------------
# AI Usage Disclosure
# --------------------------------------------------
st.markdown("---")
st.markdown(
    """
    **AI Usage Disclosure:**  
    ChatGPT was used to assist in generating Streamlit UI structure, data preprocessing
    logic, and example machine learning pipelines. Prompts included requests for
    converting IMDb rating data into a Streamlit-based predictive application.
    """
)
