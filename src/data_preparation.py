"""Data preparation service"""
import os
import re
import yaml
from sklearn.preprocessing import LabelEncoder
from database_handler import DatabaseHandler


# This class will be used to prepare data for the machine learning model.
class DataPreparation:
    """Data preparation service"""

    def __init__(self, directory):
        self.directory = directory
        self.automations = []

    # This method will be used to process the automations in the directory.
    def process_automations(self, file):
        """# Load the automations from the directory"""
        with open(file, 'r', encoding="utf-8") as stream:
            data_loaded = yaml.safe_load(stream)

        # Check if data_loaded is a list, if not, convert it to a list
        if not isinstance(data_loaded, list):
            data_loaded = [data_loaded]

        automations = []

        # Process each automation
        for automation in data_loaded:
            automation_dict = {}

            # Process triggers
            triggers = automation.get('trigger', [])
            # Check if triggers is a list, if not, convert it to a list
            if not isinstance(triggers, list):
                triggers = [triggers]
            automation_dict['triggers'] = [self.process_trigger(trigger) for trigger in triggers]

            # Process conditions
            conditions = automation.get('condition', [])
            # Check if conditions is a list, if not, convert it to a list
            if not isinstance(conditions, list):
                conditions = [conditions]
            automation_dict['conditions'] = [self.process_condition(condition) for condition in conditions]  # pylint: disable=line-too-long

            # Process actions
            actions = automation.get('action', [])
            # Check if actions is a list, if not, convert it to a list
            if not isinstance(actions, list):
                actions = [actions]
            automation_dict['actions'] = [self.process_action(action) for action in actions]

            # Process alias
            automations.append(automation_dict)

        return automations

    # This method will be used to process the triggers in the automation.
    def process_trigger(self, trigger):
        """Process a trigger dictionary and extract the 'platform', 'entity_id', and 'to' fields."""
        trigger_dict = {}
        if 'platform' in trigger:
            trigger_dict['platform'] = trigger['platform']
        if 'entity_id' in trigger:
            trigger_dict['entity_id'] = trigger['entity_id']
        if 'to' in trigger:
            trigger_dict['to'] = trigger['to']

        return trigger_dict

    # This method will be used to process the actions in the automation.
    def process_action(self, action):
        """Process an action dictionary and extract the 'service' and 'target' fields."""
        action_dict = {}
        if 'service' in action:
            action_dict['service'] = action['service']
        if 'target' in action:
            action_dict['target'] = action['target']

        return action_dict

    # This method will be used to process the conditions in the automation.
    def process_condition(self, condition):
        """Process a condition dictionary and extract the 'condition', 'before' and 'after in
        sunset/sunrise' and convert 'sunset' and 'sunrise' fields."""
        condition_dict = {}

        if 'condition' in condition:
            condition_dict['condition'] = condition['condition']
        if 'before' in condition:
            if condition['before'] in ['sunset', 'sunrise']:
                condition_dict['before'] = condition['before']
            else:
                # Check if condition['before'] is a string in the format 'HH:MM:SS'
                if isinstance(condition['before'], str):
                    condition_dict['before'] = condition['before']
                else:
                    # Convert datetime.time object into a string
                    condition_dict['before'] = condition['before'].strftime('%H:%M:%S')

        return condition_dict

    # This method will be used to process the automations in the directory.
    def correct_yaml_files(self):
        """# Loop through all the files in the directory"""
        for filename in os.listdir(self.directory):
            if filename.endswith(".yaml"):
                file = os.path.join(self.directory, filename)
                try:
                    with open(file, 'r', encoding="utf-8") as stream:
                        data = stream.read()
                        # Ensure that the quotes in the alias line are matched correctly
                        corrected_data = re.sub(r'alias:\s*"([^"]*)\'', r'alias: "\1"', data)
                    with open(file, 'w', encoding="utf-8") as output_stream:
                        output_stream.write(corrected_data)
                    print(f"File '{filename}' corrected.")
                except FileNotFoundError as file_error:  # pylint: disable=unused-variable
                    print(f"Error: File '{filename}' not found.")
                except PermissionError as permission_error:  # pylint: disable=unused-variable
                    print(f"Error: Permission denied for file '{filename}'.")
                except OSError as os_error:
                    print(f"Error accessing file '{filename}': {os_error}.")
                except Exception as other_error:  # pylint: disable=broad-except
                    print(f"Error in file '{filename}': {other_error}")

    # This method will be used to process the automations in the directory.
    def preprocess_data(self):
        """# Initialize the encoders"""
        platform_encoder = LabelEncoder()
        condition_encoder = LabelEncoder()
        service_encoder = LabelEncoder()

        # Initialize lists to store the features
        platforms = []
        conditions = []
        services = []

        # Loop through all the automations
        for automation in self.automations:
            # Loop through all the triggers, conditions, and actions
            for trigger in automation['triggers']:
                platforms.append(trigger['platform'])
            for condition in automation['conditions']:
                conditions.append(condition['condition'])
            for action in automation['actions']:
                if action is not None:  # Add this check to handle None actions
                    services.append(action['service'])

        # Fit and transform the features using the encoders
        encoded_platforms = platform_encoder.fit_transform(platforms)
        encoded_conditions = condition_encoder.fit_transform(conditions)
        encoded_services = service_encoder.fit_transform(services)

        return encoded_platforms, encoded_conditions, encoded_services

    # This method will be used to extract features from the automations.
    def feature_engineering(self):
        """# Initialize lists to store the features"""
        num_triggers = []
        num_conditions = []
        num_actions = []
        has_state_trigger = []
        has_sunrise_trigger = []

        # Loop through all the automations
        for automation in self.automations:
            # Extract features
            num_triggers.append(len(automation['triggers']))
            num_conditions.append(len(automation['conditions']))
            num_actions.append(len(automation['actions']))
            has_state_trigger.append(any(trigger['platform'] == 'state' for trigger in automation['triggers']))  # pylint: disable=line-too-long
            has_sunrise_trigger.append(any(
                trigger['platform'] == 'sun' and trigger.get('event') == 'sunrise' for trigger in
                automation['triggers']))

        # Only show the first 5 values
        return num_triggers[:5], num_conditions[:5], num_actions[:5], has_state_trigger[:5], has_sunrise_trigger[:5]  # pylint: disable=line-too-long

    # This method will be used to run all the methods in the class.
    def run_all_methods(self):
        """# Correct the YAML files"""
        self.correct_yaml_files()

        # Loop through all the files in the directory
        for filename in os.listdir(self.directory):
            if filename.endswith(".yaml"):
                file = os.path.join(self.directory, filename)
                file_automations = self.process_automations(file)
                self.automations.extend(file_automations)

        encoded_platforms, encoded_conditions, encoded_services = self.preprocess_data()

        db_handler = DatabaseHandler('automations.db')
        db_handler.create_database()
        db_handler.store_data(encoded_platforms, encoded_conditions, encoded_services)

        self.feature_engineering()
