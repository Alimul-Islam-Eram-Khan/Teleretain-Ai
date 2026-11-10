"""Train Iranian track ONLY using the Stage 2 processed Iranian records."""
from src.training_common import train_track


def run():
    return train_track("iranian")
