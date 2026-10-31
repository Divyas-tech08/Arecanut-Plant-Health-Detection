import sqlite3

DATABASE = "history.db"


def init_database():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_time TEXT NOT NULL,
            condition TEXT NOT NULL,
            condition_type TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(condition, condition_type):

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO prediction_history
        (date_time, condition, condition_type)
        VALUES
        (datetime('now', 'localtime'), ?, ?)
    """, (
        condition,
        condition_type
    ))

    connection.commit()
    connection.close()


def get_history():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            date_time,
            condition,
            condition_type
        FROM prediction_history
        ORDER BY id DESC
    """)

    history = cursor.fetchall()

    connection.close()

    return history


# ---------------------------------------------------------
# Get history for one specific condition
# ---------------------------------------------------------

def get_condition_history(condition):

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            date_time,
            condition,
            condition_type
        FROM prediction_history
        WHERE condition = ?
        ORDER BY id ASC
    """, (condition,))

    history = cursor.fetchall()

    connection.close()

    return history

def clear_history():
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("DELETE FROM prediction_history")

    connection.commit()
    connection.close()