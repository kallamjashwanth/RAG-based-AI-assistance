import pandas as pd
from src.config import DATA_PATH, TEXT_COLUMNS
from src.logger import setup_logger

logger = setup_logger()

def load_dataset():
    try:
        logger.info("Loading dataset...")
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Dataset loaded with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def preprocess_dataset(df: pd.DataFrame):
    logger.info("Preprocessing dataset...")

    # Fill missing values
    df[TEXT_COLUMNS] = df[TEXT_COLUMNS].fillna("")

    # Combine text fields
    df["combined_text"] = (
        "Title: " + df["title"] + "\n"
        "Authors: " + df["authors"] + "\n"
        "Category: " + df["primary_category"] + "\n"
        "Summary: " + df["summary"]
    )

    # Keep only useful columns
    documents = df[["entry_id", "combined_text"]].to_dict(orient="records")

    logger.info(f"Preprocessed {len(documents)} documents")
    return documents
