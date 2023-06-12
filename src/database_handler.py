"""Connectie string voor de database:"""
import sqlite3
import hashlib
import json
from typing import List, Dict, Any


class DatabaseHandler:
    """Class for handling the database"""

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.cursor: sqlite3.Cursor = None  # Add type hint for 'cursor'

    def create_database(self):
        """Create the database and the tables"""
        conn = sqlite3.connect(self.db_name)
        self.cursor = conn.cursor()

        # Create the 'raw_automations' table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_automations
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             hash TEXT UNIQUE,
             data TEXT,
             createdOn TEXT DEFAULT CURRENT_TIMESTAMP,
             updatedOn TEXT,
             deleted INTEGER DEFAULT 0)
        ''')

        # Create the 'processed_automations' table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_automations
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             platforms TEXT,
             conditions TEXT,
             services TEXT,
             createdOn TEXT DEFAULT CURRENT_TIMESTAMP,
             updatedOn TEXT,
             deleted INTEGER DEFAULT 0)
        ''')

        conn.commit()
        conn.close()

    def store_data(self, encoded_platforms: List[Any], encoded_conditions: List[Any],
                   encoded_services: List[Any]):  # pylint: disable=line-too-long
        """Store the data in the database"""
        platforms_str = str(encoded_platforms.tolist())
        conditions_str = str(encoded_conditions.tolist())
        services_str = str(encoded_services.tolist())

        conn = sqlite3.connect(self.db_name)
        self.cursor = conn.cursor()

        self.cursor.execute('''
            INSERT INTO processed_automations (platforms, conditions, services)
            VALUES (?, ?, ?)
        ''', (platforms_str, conditions_str, services_str))

        conn.commit()
        conn.close()

    def get_automation_by_id(self, automation_id) -> Dict[str, Any]:
        """Get the automation by id"""
        conn = sqlite3.connect(self.db_name)
        self.cursor = conn.cursor()

        query = "SELECT * FROM processed_automations WHERE id = ?"
        self.cursor.execute(query, (automation_id,))
        result = self.cursor.fetchone()

        if result is None:
            return None

        automation = {
            "id": result[0],
            "platforms": result[1],
            "conditions": result[2],
            "services": result[3],
            "createdOn": result[4],
            "updatedOn": result[5],
            "deleted": result[6]
        }

        return automation

    def save_to_database(self, processed_data: List[Any]):
        """Save the data to the database"""
        conn = sqlite3.connect(self.db_name)
        self.cursor = conn.cursor()

        self.cursor.execute('''
            INSERT INTO processed_automations (platforms, conditions, services)
            VALUES (?, ?, ?)
        ''', (json.dumps(processed_data[0]), json.dumps(processed_data[1]), json.dumps(processed_data[2])))  # pylint: disable=line-too-long

        conn.commit()
        conn.close()

    def store_raw_data(self, automation_data):
        """Store the raw automation data in the database, if it doesn't already exist."""
        # Calculate a hash of the automation data
        automation_hash = hashlib.sha256(json.dumps(automation_data, sort_keys=True).encode()).hexdigest()  # pylint: disable=line-too-long

        conn = sqlite3.connect(self.db_name)
        self.cursor = conn.cursor()

        # Check if an automation with the same hash already exists in the database
        self.cursor.execute('SELECT * FROM raw_automations WHERE hash = ?', (automation_hash,))
        if self.cursor.fetchone() is not None:
            # The automation already exists in the database, so don't insert it again
            print("Automation already exists in the database.")
            return

        # The automation doesn't exist in the database, so insert it
        self.cursor.execute('''
            INSERT INTO raw_automations (hash, data)
            VALUES (?, ?)
        ''', (automation_hash, json.dumps(automation_data)))

        conn.commit()
        conn.close()
