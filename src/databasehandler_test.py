"""Test the database module."""
import unittest
import json
import sqlite3
import numpy as np
from database_handler import DatabaseHandler


class TestDatabaseHandler(unittest.TestCase):
    """Test the database module."""

    def setUp(self):
        """Set up the test environment."""
        self.db_handler = DatabaseHandler('test.db')  # Use a local SQLite database for testing
        self.db_handler.create_database()

    def test_create_database(self):
        """Test that the database and the tables are created correctly."""
        # Check that the tables exist
        conn = sqlite3.connect('test.db')
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='raw_automations'")  # pylint: disable=line-too-long
        self.assertIsNotNone(cursor.fetchone())
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='processed_automations'")  # pylint: disable=line-too-long
        self.assertIsNotNone(cursor.fetchone())

        conn.close()

    def test_store_raw_data(self):
        """Test that raw automation data is stored correctly."""
        # Store some raw automation data
        self.db_handler.store_raw_data({'platform': 'state', 'condition': 'sun', 'service': 'light.turn_on'})  # pylint: disable=line-too-long

        # Check that the data was stored correctly
        conn = sqlite3.connect('test.db')
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM raw_automations LIMIT 1")
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], '{"platform": "state", "condition": "sun", "service": "light.turn_on"}')  # pylint: disable=line-too-long

        conn.close()

    def test_store_data(self):
        """Test that processed automation data is stored correctly."""
        # Store some processed automation data
        self.db_handler.store_data(np.array(['state']), np.array(['sun']), np.array(['light.turn_on']))  # pylint: disable=line-too-long

        # Check that the data was stored correctly
        conn = sqlite3.connect('test.db')
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM processed_automations WHERE id = 23")
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "['state']")
        self.assertEqual(rows[0][2], "['sun']")
        self.assertEqual(rows[0][3], "['light.turn_on']")

        conn.close()

    def test_get_automation_by_id(self):
        """Test that the correct automation is returned when getting an automation by id."""
        # Store some processed automation data
        self.db_handler.store_data(np.array(['state']), np.array(['sun']), np.array(['light.turn_on']))  # pylint: disable=line-too-long

        # Get the automation by id
        automation = self.db_handler.get_automation_by_id(24)

        # Check that the correct automation was returned
        self.assertIsNotNone(automation, "No automation found with id 24")
        self.assertEqual(automation['id'], 24)
        self.assertEqual(automation['platforms'], '"state"')
        self.assertEqual(automation['conditions'], '"sun"')
        self.assertEqual(automation['services'], '"light.turn_on"')

    def test_save_to_database(self):
        """Test that processed automation data is saved correctly."""
        # Save some processed automation data
        self.db_handler.save_to_database(['state', 'sun', 'light.turn_on'])

        # Check that the data was saved correctly
        conn = sqlite3.connect('test.db')
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM processed_automations where id = 24")
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], json.dumps("state"))
        self.assertEqual(rows[0][2], json.dumps("sun"))
        self.assertEqual(rows[0][3], json.dumps("light.turn_on"))

        conn.close()


if __name__ == '__main__':
    unittest.main()
