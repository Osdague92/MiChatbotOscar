from flask import Flask, render_template, request, jsonify
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import numpy as np
import unicodedata

# Descargar stopwords
nltk.download('stopwords')
stop_words_es = stopwords.words('spanish')

# Clase centralizada para el modelo de chatbot
class ChatModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vectorizer = CountVectorizer(stop_words=stop_words_es)
        self.encoder = LabelEncoder()
        self.model = MultinomialNB()
        self.questions = []
        self.answers = []
        self.categories = []
        self.X_vec = None
        self.y_encoded = None
        self.load_data()
        self.train_model()

    def load_data(self):
        """Carga los datos desde un archivo JSON."""
        print("Cargando datos desde:", self.data_path)
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Aplicar preprocesamiento a las preguntas
                self.questions = [self.preprocess_text(item['question']) for item in data]
                self.answers = [item['answer'] for item in data]
                self.categories = [item['category'].lower() for item in data]
            print("Datos cargados con éxito. Total preguntas:", len(self.questions))
        except FileNotFoundError:
            print(f"Error: Archivo {self.data_path} no encontrado.")
        except json.JSONDecodeError:
            print("Error: Archivo JSON malformado.")
        except Exception as e:
            print(f"Error inesperado al cargar datos: {e}")

    def train_model(self):
        """Entrena el modelo Naive Bayes."""
        print("Entrenando el modelo Naive Bayes...")
        if not self.questions:
            print("Error: No hay datos disponibles para entrenar el modelo.")
            return

        self.X_vec = self.vectorizer.fit_transform(self.questions)
        self.y_encoded = self.encoder.fit_transform(self.categories)
        self.model.fit(self.X_vec, self.y_encoded)
        print("Modelo entrenado con éxito.")

    def get_response(self, user_input):
        """Obtiene la respuesta basada en el modelo entrenado."""
        if not user_input.strip():
            return "Por favor, escribe algo."
    
        # Preprocesar la entrada del usuario antes de vectorizar
        user_input = self.preprocess_text(user_input)
    
        # Vectorizar la entrada preprocesada del usuario
        user_vec = self.vectorizer.transform([user_input])
        predicted_category_idx = self.model.predict(user_vec)[0]
        predicted_category = self.encoder.inverse_transform([predicted_category_idx])[0]
    
        # Buscar una coincidencia exacta en las preguntas
        for question, answer, category in zip(self.questions, self.answers, self.categories):
            if question == user_input:
                return answer  # Retornar la respuesta exacta
    
        # Si no hay coincidencia exacta, buscar respuestas de la categoría
        possible_answers = [
            answer for question, answer, category in zip(self.questions, self.answers, self.categories)
            if category == predicted_category
        ]
        
        if possible_answers:
            return np.random.choice(possible_answers)  # Elegir una respuesta aleatoria
    
        # Si no hay respuestas posibles
        return "Lo siento, no tengo una respuesta para tu pregunta."

    def preprocess_text(self, text):
        """Preprocesa el texto eliminando tildes, mayúsculas y stopwords."""
        text = text.lower()
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
        text = ' '.join(word for word in text.split() if word not in stop_words_es)
        return text

# Inicializar Flask y el modelo
app = Flask(__name__)
chat_model = ChatModel(data_path="data.json")

# Rutas de Flask
@app.route('/')
def index():
    """Renderiza la página principal del chatbot."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Procesa los mensajes del usuario y devuelve respuestas."""
    mensaje = request.form.get('mensaje', '').strip()
    if not mensaje:
        return jsonify(respuesta="Por favor, escribe algo.")

    try:
        respuesta = chat_model.get_response(mensaje)
        return jsonify(respuesta=respuesta)
    except Exception as e:
        return jsonify(respuesta=f"Hubo un error procesando tu mensaje: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
