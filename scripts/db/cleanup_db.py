import os
from dotenv import load_dotenv

load_dotenv()
MOOC_DB_LOCATION = os.getenv('MOOC_DB_LOCATION')

def remove_db() -> None:
    os.remove(MOOC_DB_LOCATION)
    print(f"Database at {MOOC_DB_LOCATION} removed")

