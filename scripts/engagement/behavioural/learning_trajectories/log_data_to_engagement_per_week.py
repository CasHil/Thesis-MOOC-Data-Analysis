import os
import sqlite3

from dotenv import load_dotenv
import pandas as pd

load_dotenv()

MOOC_DB_LOCATION = os.getenv('MOOC_DB_LOCATION')

def fetch_log_data() -> pd.DataFrame:
    conn = sqlite3.connect(MOOC_DB_LOCATION)

    cur = conn.cursor()

    cur.execute("""
                SELECT course_id, hash_id, week, engagement
                FROM engagement_per_week
                """)

    data = cur.fetchall()
    df = pd.DataFrame(data, columns=["Course ID", "Hash ID", "Week", "Engagement"])
    
    cur.close()
    conn.close()

    return df

def main() -> None:
    pass

if __name__ == 'main':
    main()
