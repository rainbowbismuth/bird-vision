"""
This program runs every image recognition test.
"""
from birdvision.config import configure
import birdvision.testing
import birdvision.quiet


def main():
    configure()
    birdvision.quiet.silence_tensorflow()
    birdvision.testing.run_all_tests()
    birdvision.testing.summarize_tests()


if __name__ == "__main__":
    main()
