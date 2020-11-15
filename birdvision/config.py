from dotenv import load_dotenv, find_dotenv


def configure():
    """
    Loads configuration from your .env
    """
    load_dotenv(find_dotenv())

