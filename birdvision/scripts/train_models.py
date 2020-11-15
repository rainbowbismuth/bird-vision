"""
This program trains all of our models from scratch
"""
from birdvision.character import train_small_digit, train_alpha_num
from birdvision.config import configure


def main():
    configure()
    train_small_digit()
    train_alpha_num()


if __name__ == "__main__":
    main()
