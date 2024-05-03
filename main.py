from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    try:
        tfid_x_train = tfvect.fit_transform(x_train)
        tfid_x_test = tfvect.transform(x_test)
        input_data = [news]
        vectorized_input_data = tfvect.transform(input_data)
        prediction = loaded_model.predict(vectorized_input_data)
        return prediction
    except ValueError as ve:
        return "Error: Please provide valid input data."
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            message = request.form['message']
            pred = fake_news_det(message)
            print(pred)

            fake_count = np.count_nonzero(pred == 'FAKE')
            real_count = np.count_nonzero(pred == 'REAL')

            fig = px.bar(x=['FAKE', 'REAL'], y=[fake_count, real_count],
                         labels={'x': 'Prediction', 'y': 'Count'},
                         title='Predictions Distribution')
            plot_div = fig.to_html(full_html=False)

            return render_template('result.html', prediction=pred, plot_div=plot_div)
        except Exception as e:
            return render_template('error.html', error=str(e))
    else:
        return render_template('prediction.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
