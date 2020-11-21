"""
This program trains all of our models from scratch
"""
import click

from birdvision.character import train_small_digit, train_alpha_num
from birdvision.config import configure
from birdvision.stream_state import train_stream_state


@click.command()
@click.option('--stream-state/--no-stream-state', default=False)
@click.option('--small-digit/--no-small-digit', default=False)
@click.option('--alpha-num/--no-alpha-num', default=False)
@click.option('--all/--not-all', default=False)
def train_models(stream_state, small_digit, alpha_num, all):
    if all or stream_state:
        train_stream_state()
    if all or small_digit:
        train_small_digit()
    if all or alpha_num:
        train_alpha_num()


if __name__ == "__main__":
    configure()
    train_models()
