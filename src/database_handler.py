import sqlite3


class DatabaseHandler:
    def __init__(self, db_name):
        self.db_name = db_name

    def create_database(self):
        # Connect to the SQLite database (it will be created if it doesn't exist)
        conn = sqlite3.connect(self.db_name)

        # Create a cursor object
        c = conn.cursor()

        # Create table
        c.execute('''
            CREATE TABLE IF NOT EXISTS automations
            (platforms TEXT, conditions TEXT, services TEXT)
        ''')

        # Commit the changes and close the connection
        conn.commit()
        conn.close()

    def store_data(self, encoded_platforms, encoded_conditions, encoded_services):
        # Convert the numpy arrays to lists and then to strings
        platforms_str = str(encoded_platforms.tolist())
        conditions_str = str(encoded_conditions.tolist())
        services_str = str(encoded_services.tolist())

        # Connect to the SQLite database
        conn = sqlite3.connect(self.db_name)

        # Create a cursor object
        c = conn.cursor()

        # Insert data into the table
        c.execute('''
            INSERT INTO automations (platforms, conditions, services)
            VALUES (?, ?, ?)
        ''', (platforms_str, conditions_str, services_str))

        # Commit the changes and close the connection
        conn.commit()
        conn.close()
