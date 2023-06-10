"""ML service"""
import os
import yaml
from datetime import datetime, time
import json
from sklearn.preprocessing import LabelEncoder
import re
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import numpy as np


def process_automations(file):
    with open(file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    # Check if data_loaded is a list, if not, convert it to a list
    if not isinstance(data_loaded, list):
        data_loaded = [data_loaded]

    automations = []

    for automation in data_loaded:
        automation_dict = {}

        # Process triggers
        triggers = automation.get('trigger', [])
        # Check if triggers is a list, if not, convert it to a list
        if not isinstance(triggers, list):
            triggers = [triggers]
        automation_dict['triggers'] = [process_trigger(trigger) for trigger in triggers]

        # Process conditions
        conditions = automation.get('condition', [])
        # Check if conditions is a list, if not, convert it to a list
        if not isinstance(conditions, list):
            conditions = [conditions]
        automation_dict['conditions'] = [process_condition(condition) for condition in conditions]

        # Process actions
        actions = automation.get('action', [])
        # Check if actions is a list, if not, convert it to a list
        if not isinstance(actions, list):
            actions = [actions]
        automation_dict['actions'] = [process_action(action) for action in actions]

        automations.append(automation_dict)

    return automations


def process_trigger(trigger):
    print(trigger)  # Add this line
    trigger_dict = {}

    if 'platform' in trigger:
        trigger_dict['platform'] = trigger['platform']
    if 'entity_id' in trigger:
        trigger_dict['entity_id'] = trigger['entity_id']
    if 'to' in trigger:
        trigger_dict['to'] = trigger['to']

    return trigger_dict


def process_action(action):
    action_dict = {}

    if 'service' in action:
        action_dict['service'] = action['service']
    if 'target' in action:
        action_dict['target'] = action['target']

    return action_dict


def correct_yaml_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".yaml"):
            file = os.path.join(directory, filename)
            with open(file, 'r') as stream:
                try:
                    data = stream.read()
                    # Ensure that the quotes in the alias line are matched correctly
                    corrected_data = re.sub(r'alias:\s*"([^"]*)\'', r'alias: "\1"', data)
                    with open(file, 'w') as output_stream:
                        output_stream.write(corrected_data)
                    print(f"File '{filename}' corrected.")
                except Exception as e:
                    print(f"Error in file '{filename}': {e}")


# Error in file 'automation_73.yaml': while parsing a block mapping
#   in "D:/Code/SEM7/AA_ML/automations/automation_73.yaml", line 1, column 1
# expected <block end>, but found '<scalar>'
#   in "D:/Code/SEM7/AA_ML/automations/automation_73.yaml", line 1, column 49
def prepocess_data(automations):
    # Initialize the encoders
    platform_encoder = LabelEncoder()
    condition_encoder = LabelEncoder()
    service_encoder = LabelEncoder()

    # Initialize lists to store the features
    platforms = []
    conditions = []
    services = []

    # Loop through all the automations
    for automation in automations:
        # Loop through all the triggers, conditions, and actions
        for trigger in automation['triggers']:
            platforms.append(trigger['platform'])
        for condition in automation['conditions']:
            conditions.append(condition['condition'])
        for action in automation['actions']:
            services.append(action['service'])

    # Fit and transform the features using the encoders
    encoded_platforms = platform_encoder.fit_transform(platforms)
    encoded_conditions = condition_encoder.fit_transform(conditions)
    encoded_services = service_encoder.fit_transform(services)

    return encoded_platforms, encoded_conditions, encoded_services

    # Feature engineering


# method preprocess_data
# input: list of automations
# output: list of dictionaries

def feature_engineering(automations):
    # Initialize lists to store the features
    num_triggers = []
    num_conditions = []
    num_actions = []
    has_state_trigger = []
    has_sunrise_trigger = []

    # Loop through all the automations
    for automation in automations:
        # Extract features
        num_triggers.append(len(automation['triggers']))
        num_conditions.append(len(automation['conditions']))
        num_actions.append(len(automation['actions']))
        has_state_trigger.append(any(trigger['platform'] == 'state' for trigger in automation['triggers']))
        has_sunrise_trigger.append(any(
            trigger['platform'] == 'sun' and trigger.get('event') == 'sunrise' for trigger in automation['triggers']))

    # Only show the first 5 values
    return num_triggers[:5], num_conditions[:5], num_actions[:5], has_state_trigger[:5], has_sunrise_trigger[:5]


def data_modeling(automations):
    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime.time):
                return obj.strftime('%H:%M:%S')
            return super().default(obj)

    # Convert your automations into strings
    # automation_strings = [json.dumps(automation, cls=CustomJSONEncoder) for automation in automations]
    # Add the '<end>' token to the end of each automation string
    automation_strings = [json.dumps(automation, cls=CustomJSONEncoder) + ' <end>' for automation in automations]

    # Initialize the tokenizer
    tokenizer = Tokenizer(filters='')

    # Fit the tokenizer on your automation strings
    tokenizer.fit_on_texts(automation_strings)

    # Convert your automation strings to sequences of tokens
    sequences = tokenizer.texts_to_sequences(automation_strings)

    # Pad your sequences so they all have the same length
    sequences = pad_sequences(sequences, padding='post')

    # Create your input and target sequences
    X = sequences[:, :-1]
    y = sequences[:, 1:]

    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=X.shape[1]))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    # Reshape y to be 3-dimensional, as the model expects
    y_reshaped = y.reshape(*y.shape, 1)

    # Train the model
    model.fit(X, y_reshaped, epochs=10, validation_split=0.2)
    model.save('model.h5')

    max_length = 20
    # Define the starting sequence
    start_sequence = '{"alias": "Example automation", "trigger": {"platform": "state", "entity_id": "sun.sun", ' \
                     '"to": "below_horizon"}, "condition": {"condition": "state", "entity_id": ' \
                     '"device_tracker.person1", "state": "home"}, "action": {"service": "light.turn_on", "target": {' \
                     '"entity_id": "light.living_room"}}'

    # Convert the start sequence to tokens
    sequence = tokenizer.texts_to_sequences([start_sequence])[0]

    # Pad the sequence
    sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

    # Use the model to predict the next token
    prediction = model.predict(sequence)
    predicted_token = np.argmax(prediction[0, -1, :])

    # Add the predicted token to the sequence
    sequence = np.append(sequence[0], predicted_token)

    # Continue predicting tokens until the end token is predicted or the maximum length is reached
    while predicted_token != tokenizer.word_index['<end>'] and len(sequence) < max_length:
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        prediction = model.predict(sequence)
        predicted_token = np.argmax(prediction[0, -1, :])
        sequence = np.append(sequence[0], predicted_token)

    # Convert the sequence of tokens back into text
    automation = tokenizer.sequences_to_texts([sequence])

    return automation[0]


# data modeling
def run_all_methods(directory):
    automations = []

    correct_yaml_files(directory)

    # loop through all the yaml files in the directory D:/Temp/yaml
    for filename in os.listdir(directory):
        if filename.endswith(".yaml"):
            print("Processing file: " + filename)
            file = os.path.join(directory, filename)
            file_automations = process_automations(file)
            automations.extend(file_automations)

    # preprocess_data(automations)
    print(prepocess_data(automations))

    # feature_engineering(automations)
    print(feature_engineering(automations))

    print(data_modeling(automations))


def process_condition(condition):
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


if __name__ == '__main__':
    directory = 'D:/Code/SEM7/AA_ML/automations/'
    directory = 'D:/Temp/yaml'
    automations = []

    correct_yaml_files(directory)

    # loop through all the yaml files in the directory D:/Temp/yaml
    for filename in os.listdir(directory):
        if filename.endswith(".yaml"):
            print("Processing file: " + filename)
            file = os.path.join(directory, filename)
            file_automations = process_automations(file)
            automations.extend(file_automations)

    # preprocess_data(automations)
    print(prepocess_data(automations))

    # feature_engineering(automations)
    print(feature_engineering(automations))

    print(data_modeling(automations))
