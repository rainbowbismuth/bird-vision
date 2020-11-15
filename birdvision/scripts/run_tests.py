"""
This program runs every image recognition test.
"""
import birdvision.quiet
import birdvision.testing
from birdvision.config import configure


def main():
    configure()
    birdvision.quiet.silence_tensorflow()
    birdvision.testing.run_all_tests()
    birdvision.testing.summarize_tests()


if __name__ == "__main__":
    main()
