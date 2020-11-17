"""
This program trains all of our models from scratch
"""
from birdvision.character import train_small_digit, train_alpha_num
from birdvision.stream_state import train_stream_state
from birdvision.config import configure

import click


@click.command()
@click.option('--stream-state/--no-stream-state', default=False)
@click.option('--small-digit/--no-small-digit', default=False)
@click.option('--alpha-num/--no-alpha-num', default=False)
def train_models(stream_state, small_digit, alpha_num):
    if stream_state:
        train_stream_state()
    if small_digit:
        train_small_digit()
    if alpha_num:
        train_alpha_num()


if __name__ == "__main__":
    configure()
    train_models()
