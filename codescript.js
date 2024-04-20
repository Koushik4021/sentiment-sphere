function getEmotionalStatus() {
    var inputText = document.getElementById('textInput').value;
    var emotionalStatus = analyzeSentiment(inputText);
    displayEmotionalStatus(emotionalStatus);
}

function analyzeSentiment(text) {
    var words = text.toLowerCase().split(/\s+/);
    var emotionalWords = {
        "sad": ["sad", "depressed", "melancholy", "unhappy", "lonely","kill","die","lost"],
        "happy": ["happy", "euphoric", "joyful", "cheerful"],
        "neutral": ["fine", "alright", "okay","meh"]
    };
    var counts = {};
    Object.keys(emotionalWords).forEach(function(emotion) {
        counts[emotion] = 0;
        emotionalWords[emotion].forEach(function(word) {
            counts[emotion] += words.filter(function(w) { return w === word; }).length;
        });
    });
    var maxEmotion = Object.keys(counts).reduce(function(a, b) { return counts[a] > counts[b] ? a : b; });
    return maxEmotion;
}

function displayEmotionalStatus(emotion) {
    var statusElement = document.getElementById('emotionalStatus');
    var emoji;
    var message;

    switch (emotion) {
        case 'sad':
            emoji = '😢\n';
            message = "You seem sad. Don't worry, everything will be okay!";
            break;
        case 'happy':
            emoji = '😄\n';
            message = "You seem happy! Keep spreading the joy!";
            break;
        case 'neutral':
            emoji = '😐\n';
            message = "You seem neutral. That's okay too!";
            break;
        default:
            emoji = '🤔\n';
            message = "Hmm... I'm not sure how you're feeling.";
            break;
    }

    statusElement.innerHTML = `<p>${emoji} ${message}</p>`;
}
