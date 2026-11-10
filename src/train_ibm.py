"""Train IBM benchmark ONLY using the Stage 2 processed IBM records."""
from src.training_common import train_track


def run():
    return train_track("ibm")
