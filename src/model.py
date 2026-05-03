import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class ClinicalModel:

    def __init__(self):
        self.columns = [
            "Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigree","Age"
        ]
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.train()

    def load_data(self):
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        df = pd.read_csv(url, header=None)

        df.columns = self.columns + ["Outcome"]
        return df

    def train(self):
        df = self.load_data()

        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

    def predict(self, input_data):
        df = pd.DataFrame([input_data], columns=self.columns)
        return int(self.model.predict(df)[0])

    def feature_importance(self):
        return dict(zip(self.columns, self.model.feature_importances_))