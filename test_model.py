import unittest

# import python file
from main import fit_and_score
from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class TestFitAndScore(unittest.TestCase):
    def test_fit_and_score(self):
        # Load a simple test dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define a simple model
        models = {"Logistic Regression": LogisticRegression(n_jobs=-1, max_iter=10000)}

        # Call fit_and_score and get the result
        result = fit_and_score(models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        # Check that the result is as expected
        self.assertIn('Logistic Regression', result)  # Check if 'Logistic Regression' is a key in the result dict


if __name__ == "__main__":
    unittest.main()
