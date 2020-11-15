"""
This program runs every image recognition test.
"""
from dotenv import load_dotenv, find_dotenv
import birdvision.testing
import birdvision.quiet


def main():
    load_dotenv(find_dotenv())
    birdvision.quiet.silence_tensorflow()
    birdvision.testing.run_all_tests()


if __name__ == "__main__":
    main()
