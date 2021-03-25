import os
from pathlib import Path, PurePath

relative_path = PurePath(__file__).parent.relative_to(Path.cwd())

# models path
MODELS_PATH = relative_path / 'external/'
AUTOREC_MODEL = MODELS_PATH / "autorec"
SENTIMENT_ANALYSIS_MODEL = MODELS_PATH /  'sentiment_analysis'
RECOMMENDER_MODEL = MODELS_PATH /  "recommender"
# data set path
REDIAL_DATA_PATH = relative_path / 'external/redial'
TRAIN_PATH = "train_data"
VALID_PATH = "valid_data"
TEST_PATH = "test_data"

MOVIE_PATH = os.path.join(REDIAL_DATA_PATH, "movies_merged.csv")
VOCAB_PATH = os.path.join(REDIAL_DATA_PATH, "vocabulary.pkl")

# reddit data path
# (If you want to pre-train the model on the movie subreddit, from the FB movie dialog dataset)
# Note: this was not used to produce the results in "Towards Deep Conversational Recommendations" as it did not
# produce good results for us.
REDDIT_PATH = "/path/to/fb_movie_dialog_dataset/task4_reddit"
REDDIT_TRAIN_PATH = "task4_reddit_train.txt"
REDDIT_VALID_PATH = "task4_reddit_dev.txt"
REDDIT_TEST_PATH = "task4_reddit_test.txt"

CONVERSATION_LENGTH_LIMIT = 40  # conversations are truncated after 40 utterances
UTTERANCE_LENGTH_LIMIT = 80  # utterances are truncated after 80 words

# Movielens ratings path
ML_DATA_PATH = relative_path / "external/movielens"
ML_SPLIT_PATHS = [os.path.join(ML_DATA_PATH, "split0"),
                  os.path.join(ML_DATA_PATH, "split1"),
                  os.path.join(ML_DATA_PATH, "split2"),
                  os.path.join(ML_DATA_PATH, "split3"),
                  os.path.join(ML_DATA_PATH, "split4"), ]
ML_TRAIN_PATH = "train_data"
ML_VALID_PATH = "valid_data"
ML_TEST_PATH = "test_data"