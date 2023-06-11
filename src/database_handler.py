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

    def get_automation_by_id(self, automation_id):
        query = "SELECT * FROM automations WHERE id = ? AND deleted = 0"
        self.cursor.execute(query, (automation_id,))
        result = self.cursor.fetchone()

        if result is None:
            return None

        automation = {
            "id": result[0],
            "value": result[1],
            "type": result[2],
            "createdOn": result[3],
            "updatedOn": result[4],
            "deleted": result[5]
        }

        return automation.values()

    def save_to_database(self, processed_data):
        # Connect to the SQLite database
        conn = sqlite3.connect(self.db_name)

        # Create a cursor object
        c = conn.cursor()

        # Insert data into the table
        c.execute('''
            INSERT INTO automations (platforms, conditions, services)
            VALUES (?, ?, ?)
        ''', (processed_data[0], processed_data[1], processed_data[2]))

        # Commit the changes and close the connection
        conn.commit()
        conn.close()