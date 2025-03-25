from database_connection import get_connection


def test_db_connection():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT version();")
    print("PostgreSQL version:", cur.fetchone())
    cur.close()
    conn.close()


if __name__ == "__main__":
    test_db_connection()
