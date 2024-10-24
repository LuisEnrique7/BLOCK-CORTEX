from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from chatbot import Chatbot

app = Flask(__name__)
CORS(app)

chatbot = Chatbot()

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')  # Esto busca el index.html en el mismo directorio que app.py

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data['message']
    response = chatbot.get_response(user_message)
    return jsonify({'reply': response})  # Cambié 'response' a 'reply' para que coincida con tu frontend

if __name__ == '__main__':
    app.run(debug=True)
