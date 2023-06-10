import json
from datetime import datetime
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import numpy as np

class MachineLearning:
    def __init__(self, automations):
        self.automations = automations

    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime.time):
                return obj.strftime('%H:%M:%S')
            return super().default(obj)

    def data_modeling(self):
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime.time):
                    return obj.strftime('%H:%M:%S')
                return super().default(obj)

        # Convert your automations into strings
        # automation_strings = [json.dumps(automation, cls=CustomJSONEncoder) for automation in automations]
        # Add the '<end>' token to the end of each automation string
        automation_strings = [json.dumps(automation, cls=self.CustomJSONEncoder) + ' <end>' for automation in self.automations]

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

        max_length = 16
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