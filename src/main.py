from data_preparation import DataPreparation
from machine_learning import MachineLearning

if __name__ == '__main__':
    directory = 'D:/Temp/yaml'
    # location in the project where the automations are stored
    # src/automations
    # directory = 'automations'
    print(f"Directory: {directory}")

    data_prep = DataPreparation(directory)
    data_prep.run_all_methods()

    ml = MachineLearning(data_prep.automations)  # Pass the automations variable to MachineLearning
    print(ml.data_modeling())
