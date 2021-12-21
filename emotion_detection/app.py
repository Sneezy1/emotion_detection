import pandas as pd
from flask import Flask, request, render_template
from model import get_prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/emotion-detection',methods=['POST'])
def predict():

	input_features = request.form.get('text')
	output = get_prediction(input_features)
	result_df = pd.DataFrame(columns=['label','score'])
	
	for i in range(len(output[0])):
		new_row = pd.DataFrame(data=output[0][i],index=[i])
		result_df = result_df.append(new_row)
	html_table = result_df.to_html(index=False)
	
	return render_template('index.html', prediction_text='Results for every emotion:{}'.format(html_table))

if __name__ == "__main__":
    app.run(debug=True)