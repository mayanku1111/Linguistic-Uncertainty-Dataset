class Accuracy:
    def __init__(self):
        pass

    def evaluate(self, responses_df):
        # responses_df contains column accuracy, which is a 0/1 value
        return responses_df["accuracy"].mean()
