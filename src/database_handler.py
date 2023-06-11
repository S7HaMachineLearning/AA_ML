"""Connectie string voor de database:"""
import sqlite3
from typing import List, Dict, Any


class DatabaseHandler:
    """Class for handling the database"""
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.cursor: sqlite3.Cursor = None  # Add type hint for 'cursor'

    def create_database(self):
        conn = sqlite3.connect(self.db_name)
        self.cursor = conn.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS automations
            (platforms TEXT, conditions TEXT, services TEXT)
        ''')

        conn.commit()
        conn.close()

    def store_data(self, encoded_platforms: List[Any], encoded_conditions: List[Any], encoded_services: List[Any]):
        platforms_str = str(encoded_platforms.tolist())
        conditions_str = str(encoded_conditions.tolist())
        services_str = str(encoded_services.tolist())

        conn = sqlite3.connect(self.db_name)
        self.cursor = conn.cursor()

        self.cursor.execute('''
            INSERT INTO automations (platforms, conditions, services)
            VALUES (?, ?, ?)
        ''', (platforms_str, conditions_str, services_str))

        conn.commit()
        conn.close()

    def get_automation_by_id(self, automation_id) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_name)
        self.cursor = conn.cursor()

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

        return automation

    def save_to_database(self, processed_data: List[Any]):
        conn = sqlite3.connect(self.db_name)
        self.cursor = conn.cursor()

        self.cursor.execute('''
            INSERT INTO automations (platforms, conditions, services)
            VALUES (?, ?, ?)
        ''', (processed_data[0], processed_data[1], processed_data[2]))

        conn.commit()
        conn.close()
