# predictor/train_model.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

def train_and_save_model():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    joblib.dump(model, 'predictor/model.pkl')

if __name__ == '__main__':
    train_and_save_model()