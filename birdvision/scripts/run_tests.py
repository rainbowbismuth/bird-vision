"""
This program runs every image recognition test.
"""
import birdvision.quiet
import birdvision.testing
from birdvision.config import configure


def main():
    configure()
    birdvision.quiet.silence_tensorflow()
    test_framework = birdvision.testing.run_all_tests()
    test_framework.summarize_to_stdout()


if __name__ == "__main__":
    main()
