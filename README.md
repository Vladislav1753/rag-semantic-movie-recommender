# Semantic Movie Recommender

A retrieval-augmented web app that recommends movies based on semantic similarity between user input and movie overviews.

## About the project

This is my 2d RAG-powered pet project  — a simple web app that recommends movies based on their descriptions. Instead of filtering by hard-coded genres or titles, users can search semantically: describe the kind of movie they're in the mood for using natural language — like “eerie and violent”, “uplifting and emotional”, or “set in space, mysterious” — and the system will try to match it based on the movie overviews.

The goal is to provide more personalized, vibe-based movie recommendations — something that goes beyond standard filtering by genre or rating. It works especially well when you don’t know what exactly you’re looking for but can describe what you feel like watching.

However, the project also faces challenges, primarily due to the nature of the data itself. Movie overviews are usually written by different people, with varying styles, focus, and depth. Some overviews may briefly summarize the plot, others may emphasize the atmosphere or tone, and some might highlight key themes without detailed storytelling. This inconsistency can affect the quality and accuracy of semantic search results, as the embeddings and similarity comparisons rely heavily on the textual data’s expressiveness and uniformity.

Despite these challenges, the project demonstrates how retrieval-augmented generation (RAG) and semantic embeddings can be applied to creative recommendation systems, and provides a solid foundation for further improvements such as dataset standardization, fine-tuning embedding models, or incorporating user feedback.




## Features
- **Semantic search powered by text embeddings** — users can describe what they want to watch in free-form language, and the app finds relevant movies based on meaning, not just keywords.

- **Emotion-based filtering** — each movie overview is analyzed with a Hugging Face model to detect emotional tones like joy, sadness, fear, etc., enabling mood-based recommendations.

- **Rich filtering options** — filter movies by genre, language, runtime, emotional tone, and release year via a convenient Gradio-based UI.

- **Fast and scalable retrieval** — movie descriptions are stored in a vector database (Chroma) for quick semantic similarity search.

- **Modular pipeline** — data preparation, emotion tagging, and vector DB creation are separated into clear Jupyter notebooks, making it easy to improve or extend the project.

- **Deployed with Google Cloud Run** — the app is publicly accessible without local setup.

- **Dockerized environment** — clean, reproducible deployment using Docker.

## Web Interface
The front end is built with Gradio, allowing users to interact with the recommender through a clean and simple UI.

<!-- You can change to the actual path or keep as is for now -->

### Filter Options

- **Movie description** – A free-text field for semantic search (main driver of recommendations)
- **Emotional tone** – Optional filter based on emotion classification of each movie overview
- **Language** – Filter by original movie language
- **Min/Max runtime** – Filter by movie length (in minutes)
- **Genres** – Multi-select from 19 common genres
- **Start/End year** – Filter by release year

## Emotion Classification

Each movie overview is also analyzed for emotional tone using a pre-trained Hugging Face model:

```python
classifier = pipeline("text-classification",
                      model="j-hartmann/emotion-english-distilroberta-base",
                      top_k=None,
                      device=0)
```

The classifier detects the presence of the following emotions in the description:

- `anger`
- `disgust`
- `fear`
- `joy`
- `sadness`
- `surprise`
- `neutral`

These labels are used as an optional filter in the app, letting users find movies that match a specific emotional vibe (e.g., joyful, sad, or suspenseful).

## Key Technologies

- Python
- Gradio – UI for interacting with the recommender
- LangChain – For document loading and text splitting
- Chroma – Vector database for semantic retrieval
- Sentence Transformers / OpenAI / Hugging Face – For generating text embeddings
- Hugging Face pipeline – For emotion classification

## Future Work

- Improve quality of overview data with preprocessing or summarization
- Add user feedback loop for ranking or fine-tuning
- Explore fine-tuned embedding models for better semantic matching
- Add collaborative filtering signals (optional hybrid approach)


## Project Files

Here's a quick overview of the main files in the repository:

- `data_preparation.ipynb` – Prepares and cleans the raw movie data.
- `vector_db.ipynb` – Builds a vector database from the processed dataset using semantic embeddings.
- `semantic_analysis.ipynb` – Performs emotional tone classification on movie overviews using a Hugging Face model with 7 emotion categories.
- `app.py` – The main application script built with Gradio. Handles UI and integrates all logic for filtering and recommendations.
- `id_description_keywords_genres.txt` – Text file containing overviews combined with keywords and genres that are uploaded to vector database.
- `movies_cleaned.csv` – Cleaned version of the dataset after preprocessing.
- `movies_with_emotions.csv` – Dataset with added emotion labels for each movie overview.
- `requirements.txt` – List of Python dependencies.
- `Dockerfile` – Configuration for containerizing the application.
- `.env` – Environment variables (not included in repo, used locally).
