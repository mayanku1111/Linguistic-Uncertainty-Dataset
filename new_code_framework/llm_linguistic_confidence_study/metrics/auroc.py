from sklearn.metrics import roc_auc_score

class AUROC:
    def __init__(self):
        pass

    def evaluate(self, responses_df):
        # responses_df contains column confidence, which is a float value
        return roc_auc_score(responses_df["accuracy"], responses_df["confidence"])