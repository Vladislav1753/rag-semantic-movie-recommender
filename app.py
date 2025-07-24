import gradio as gr
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import ast

load_dotenv()

movies = pd.read_csv("movies_with_emotions.csv")
movies["release_date"] = pd.to_datetime(movies["release_date"])

movies["genre_list"] = movies["genre_list"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
all_genres = [genre for sublist in movies["genre_list"] for genre in sublist]
unique_genres = sorted(set(all_genres))

db_movies = Chroma(
    persist_directory="chroma_db_movies",
    embedding_function=OpenAIEmbeddings()
)


def retrieve_semantic_recommendations(
        query: str,
        language: str,
        tone: str,
        min_runtime: int,
        max_runtime: int,
        start_year: str,
        end_year: str,
        genres: list[str],
        initial_top_k: int = 100,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_movies.similarity_search(query, k=initial_top_k)
    movies_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    movie_recs = movies[movies["id"].isin(movies_list)].copy()

    if language != "All":
        movie_recs = movie_recs[movie_recs["original_language"] == language]

    movie_recs = movie_recs[
        (movie_recs["runtime"] >= min_runtime) & (movie_recs["runtime"] <= max_runtime)
        ]

    movie_recs = movie_recs[
        (movie_recs["release_date"].dt.year >= start_year) &
        (movie_recs["release_date"].dt.year <= end_year)
        ]

    if genres:
        movie_recs = movie_recs[
            movie_recs["genre_list"].apply(lambda g_list: any(genre in g_list for genre in genres))
        ]

    if tone == "Happy":
        movie_recs = movie_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        movie_recs = movie_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        movie_recs = movie_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        movie_recs = movie_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        movie_recs = movie_recs.sort_values(by="sadness", ascending=False)

    return movie_recs.head(final_top_k)


def recommend_movies(
        query: str,
        language: str,
        tone: str,
        min_runtime: int,
        max_runtime: int,
        start_year: str,
        end_year: str,
        genres: list[str]
):
    recommendations = retrieve_semantic_recommendations(
        query=query,
        language=language,
        tone=tone,
        min_runtime=min_runtime,
        max_runtime=max_runtime,
        start_year=start_year,
        end_year=end_year,
        genres=genres
    )

    results = []
    for _, row in recommendations.iterrows():
        desc = row["overview"]
        short_desc = " ".join(desc.split()[:30]) + "..."
        caption = f"{row['title']}: {short_desc}"
        results.append((row["poster_url"], caption))

    return results


# ---------------- UI ---------------- #

tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
languages = ["All"] + sorted(movies["original_language"].dropna().unique().tolist())

min_runtime_val = int(movies["runtime"].min())
max_runtime_val = int(movies["runtime"].max())
min_date_val = str(movies["release_date"].min().date())
max_date_val = str(movies["release_date"].max().date())

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 🎬 Semantic Movie Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Movie description", placeholder="e.g., A story about hope and resilience")

    with gr.Row():
        tone_dropdown = gr.Dropdown(choices=tones, label="Emotional tone", value="All")
        lang_dropdown = gr.Dropdown(choices=languages, label="Language", value="All")

    with gr.Row():
        min_runtime = gr.Slider(min_runtime_val, max_runtime_val, value=0, step=1, label="Min runtime (min)")
        max_runtime = gr.Slider(min_runtime_val, max_runtime_val, value=242, step=1, label="Max runtime (min)")


    with gr.Row():
        genre_filter = gr.CheckboxGroup(
        choices=unique_genres,
        label="Filter by genres:",
        value=[],
        info="You can select multiple genres"
    )

    with gr.Row():
        release_start = gr.Number(label="Start year", value=int(min_date_val[:4]), precision=0)
        release_end = gr.Number(label="End year", value=int(max_date_val[:4]), precision=0)

    submit_button = gr.Button("🔍 Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Movies", columns=4, rows=2)

    submit_button.click(
        fn=recommend_movies,
        inputs=[
            user_query, lang_dropdown, tone_dropdown,
            min_runtime, max_runtime,
            release_start, release_end,
            genre_filter
        ],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch()
