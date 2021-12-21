import pandas as pd
from transformers import pipeline

classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

def get_prediction(input_features):
	prediction = classifier(input_features)
	return prediction
