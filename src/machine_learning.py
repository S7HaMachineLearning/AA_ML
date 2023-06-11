""" A module for machine learning"""
import json
import pickle
from datetime import time
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.models import load_model


class MachineLearning:
    """A class for machine learning"""

    def __init__(self, automations=None, max_length=20, ):
        self.automations = automations
        self.model = None  # The machine learning model
        self.tokenizer = Tokenizer(filters='')  # Initialize the tokenizer
        self.max_length = max_length  # The maximum length of a sequence

    class CustomJSONEncoder(json.JSONEncoder):
        """A custom JSON encoder"""
        def default(self, object):  # pylint: disable=redefined-builtin
            if isinstance(object, time):
                return object.strftime('%H:%M:%S')
            return super().default(object)

    # method to load the model and tokenizer
    def load_model(self, model_path, tokenizer_path):  # Add the tokenizer_path parameter
        """ Load the model from the specified path """
        self.model = load_model(model_path)
        # Load the tokenizer from the specified path
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    # method to save model and tokenizer
    def save_model(self, model_path, tokenizer_path):  # Add the tokenizer_path parameter
        """# Save the model to the specified path"""
        self.model.save(model_path)
        # Save the tokenizer to the specified path
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # method to model the data and save the model
    def data_modeling(self, automations):  # pylint: disable=unused-argument
        """# Convert your automations into strings"""
        # Add the '<end>' token to the end of each automation string
        automation_strings = [json.dumps(automation, cls=self.CustomJSONEncoder) + ' <end>' for automation in  # pylint: disable=line-too-long
                              self.automations]

        # Initialize the tokenizer
        tokenizer = Tokenizer(filters='')

        # Fit the tokenizer on your automation strings
        tokenizer.fit_on_texts(automation_strings)

        # Convert your automation strings to sequences of tokens
        sequences = tokenizer.texts_to_sequences(automation_strings)

        # Pad your sequences so they all have the same length
        sequences = pad_sequences(sequences, padding='post')

        # Create your input and target sequences
        x_axis = sequences[:, :-1]
        y_axis = sequences[:, 1:]

        # Define the model
        model = Sequential()
        model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=x_axis.shape[1]))  # pylint: disable=line-too-long
        model.add(LSTM(64, return_sequences=True))
        model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        # Reshape y to be 3-dimensional, as the model expects
        y_reshaped = y_axis.reshape(*y_axis.shape, 1)

        # Train the model
        model.fit(x_axis, y_reshaped, epochs=10, validation_split=0.2)
        model.save('model.h5')

        max_length = 16
        # Define the starting sequence
        start_sequence = '{"alias": "Example automation", "trigger": {"platform": "state", "entity_id": "sun.sun", "to": "below_horizon"}, "condition": {"condition": "state", "entity_id": "device_tracker.person1", "state": "home"}, "action": {"service": "light.turn_on", "target": {"entity_id": "light.living_room"}}}'  # pylint: disable=line-too-long

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

    # method to generate automation
    def generate_automation(self, start_sequence):
        """# Convert the start sequence to tokens"""
        sequence = self.tokenizer.texts_to_sequences([start_sequence])[0]

        # Pad the sequence
        sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')

        # Use the model to predict the next token
        prediction = self.model.predict(sequence)
        predicted_token = np.argmax(prediction[0, -1, :])

        # Add the predicted token to the sequence
        sequence = np.append(sequence[0], predicted_token)

        # Continue predicting tokens until the end token is predicted or the maximum length is reached
        while predicted_token != self.tokenizer.word_index['<end>'] and len(sequence) < self.max_length:
            sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')
            prediction = self.model.predict(sequence)
            predicted_token = np.argmax(prediction[0, -1, :])
            sequence = np.append(sequence[0], predicted_token)

        # Convert the sequence of tokens back into text
        automation = self.tokenizer.sequences_to_texts([sequence])

        return automation[0]
