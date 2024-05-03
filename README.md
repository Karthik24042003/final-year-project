# Fake News Detection Web App

This is a Flask web application for detecting fake news using a Passive Aggressive Classifier trained on TF-IDF vectors.

## Description

This web application allows users to input news text, predicts whether it's fake or real, and displays the prediction along with a distribution plot of predictions.

## Prerequisites

- Python 3.x
- Flask
- scikit-learn
- pandas
- plotly

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/your_username/fake-news-detection-app.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Access the application in your web browser at `http://127.0.0.1:5000/`.

3. Enter the news text you want to check and submit the form.

4. The application will predict whether the news is fake or real and display the result along with a distribution plot.

## Files

- `app.py`: Flask application code.
- `model.pkl`: Pickled trained model for fake news detection.
- `news.csv`: Dataset containing news text and labels.
- `templates/`: HTML templates for the web application.
- `static/`: Static files (e.g., CSS, JavaScript) for the web application.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Inspiration: This project was inspired by the need to combat misinformation and fake news.
s