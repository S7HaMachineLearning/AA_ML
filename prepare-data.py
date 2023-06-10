"""Data preparation service"""
from datetime import datetime
import sqlite3
import models


# This class will be used to prepare data for the machine learning model.
class DataPreparation:

    # This method will be used to get the data from the YAML file.
    def get_data(self, yaml_file):
        """Get data from YAML file."""
        try:
            # Read YAML file
            with open(yaml_file) as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
                return data
        except yaml.YAMLError as err:
            print(err)
            return None

