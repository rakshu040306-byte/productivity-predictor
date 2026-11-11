import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class ProductivityPredictor:
    def __init__(self, data_path="data/usage_logs.csv"):
        self.data_path = data_path
        self.model = DecisionTreeClassifier()
        self.train_model()

    def train_model(self):
        df = pd.read_csv(self.data_path)
        X = df[["social_media_mins", "study_app_mins", "hour_of_day"]]
        y = df["is_productive"]
        self.model.fit(X, y)

    def predict(self, social_mins: int, study_mins: int, hour: int):
        input_data = pd.DataFrame(
            [[social_mins, study_mins, hour]],
            columns=["social_media_mins", "study_app_mins", "hour_of_day"]
        )
        prediction = self.model.predict(input_data)[0]
        return "Productive" if prediction == 1 else "High Distraction Risk"
