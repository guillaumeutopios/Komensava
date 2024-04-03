from flask import Flask, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from deep_translator import GoogleTranslator

app = Flask(__name__)

@app.route('/komensava', methods=['POST'])
def komensava():
    # Récupérer le texte à analyser depuis la requête POST
    data = request.get_json()
    text = data['text']

    # Traduire en anglais depuis n'importe quelle langue
    text = GoogleTranslator(source='auto', target='en').translate(text)

    # Analyser le sentiment du texte en utilisant NLTK
    nltk.download('vader_lexicon')
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)

    # Déterminer le sentiment dominant
    sentiment = 'positif' if sentiment_scores['compound'] > 0.1 else 'négatif' if sentiment_scores['compound'] < -0.1 else 'neutre'

    # Créer la réponse de l'API
    response = {
        'Texte en anglais': text,
        'Valeur de sentiment': sentiment_scores['compound'],
        'Sentiment': sentiment
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)