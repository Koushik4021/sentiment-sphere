from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Define the responses for different emotional states
responses = {
    "sad": ["I'm sorry to hear that you're feeling sad.",
            "It's okay to feel sad sometimes. How can I help?",
            "If you want to talk about it, I'm here for you."],
    "happy": ["That's great to hear that you're feeling happy!",
              "Happiness is contagious! Keep spreading those positive vibes.",
              "I'm glad to hear that you're in a good mood."],
    "neutral": ["It seems like you're feeling neutral about things. That's okay.",
                "Sometimes it's normal to feel neither happy nor sad.",
                "If you ever want to talk or need support, feel free to reach out."],
}

# Define emotional words for different emotions including their variations
emotional_words = {
    "sad": ["sad", "depressed", "melancholy", "unhappy", "lonely"],
    "happy": ["happy", "euphoric", "joyful", "cheerful"],
    "neutral": ["fine", "alright", "okay"],
}

# Tokenize and lemmatize the text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token)
        lemmatized_tokens.append(lemma)
    return lemmatized_tokens

# Identify the emotional words in the text
def identify_emotions(text):
    emotions = {}
    lemmatizer = WordNetLemmatizer()
    tagged_words = nltk.pos_tag(text)
    for word, tag in tagged_words:
        if tag.startswith('J'):  # Adjective
            lemma = lemmatizer.lemmatize(word, pos='a')  # Lemmatize as adjective
            for emotion, words in emotional_words.items():
                if lemma in words:
                    emotions[emotion] = emotions.get(emotion, 0) + 1
        elif tag.startswith('R'):  # Adverb
            lemma = lemmatizer.lemmatize(word, pos='r')  # Lemmatize as adverb
            for emotion, words in emotional_words.items():
                if lemma in words:
                    emotions[emotion] = emotions.get(emotion, 0) + 1
        elif tag.startswith('V'):  # Verb
            lemma = lemmatizer.lemmatize(word, pos='v')  # Lemmatize as verb
            for emotion, words in emotional_words.items():
                if lemma in words:
                    emotions[emotion] = emotions.get(emotion, 0) + 1
        else:
            lemma = lemmatizer.lemmatize(word)  # Default lemmatization
            for emotion, words in emotional_words.items():
                if lemma in words:
                    emotions[emotion] = emotions.get(emotion, 0) + 1
    return emotions

# Calculate percentage of each emotion in the text
def calculate_percentage(emotions):
    total_words = sum(emotions.values())
    percentages = {emotion: (count / total_words) * 100 if total_words != 0 else 0 for emotion, count in emotions.items()}
    return percentages

# Determine the response based on the highest percentage of emotion
def determine_response(percentages):
    if not percentages:
        return "I'm sorry, I couldn't understand your emotions properly. Could you express a little more?"
    
    max_emotion = max(percentages, key=percentages.get)
    if max_emotion == "neutral":
        return "It seems like you're feeling neutral about things. If you'd like to share more, feel free to do so."
    else:
        return responses[max_emotion][0]  # Selecting the first response for simplicity

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.json['input']
    preprocessed_input = preprocess_text(user_input)
    identified_emotions = identify_emotions(preprocessed_input)
    percentages = calculate_percentage(identified_emotions)
    response = determine_response(percentages)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
