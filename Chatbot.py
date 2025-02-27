import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import json
import nltk
from nltk.corpus import stopwords

# Descargar las stop words si no están disponibles
nltk.download('stopwords')
stop_words_es = stopwords.words('spanish')

class ChatModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vectorizer = CountVectorizer(stop_words=stop_words_es)
        self.model = MultinomialNB()  # Usamos Naive Bayes
        self.label_encoder = LabelEncoder()  # Para las categorías
        self.questions = []
        self.answers = []
        self.categories = []  # Aquí guardamos las categorías
        self.load_data()
        self.train_model()

    def load_data(self):
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Cargar preguntas, respuestas y categorías
                self.questions = [item['question'].lower() for item in data]
                self.answers = [item['answer'] for item in data]
                self.categories = [item['category'] for item in data]  # Asumiendo que cada entrada tiene una categoría
        except FileNotFoundError:
            print(f"Error: Archivo {self.data_path} no encontrado.")
        except json.JSONDecodeError:
            print("Error: Archivo JSON malformado.")
        except Exception as e:
            print(f"Error inesperado al cargar datos: {e}")

    def train_model(self):
        # Transformamos las preguntas a un formato numérico
        if not self.questions:
            print("Error: No hay preguntas disponibles para entrenar el modelo.")
            return
        
        X_vec = self.vectorizer.fit_transform(self.questions)
        # Usamos las categorías convertidas a números con LabelEncoder
        y = self.label_encoder.fit_transform(self.categories)
      
        # Definir los parámetros para GridSearch
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0],  # Valores posibles para alpha
            'fit_prior': [True, False]  # Si se aprende la probabilidad a priori
        }

        # Usar GridSearchCV para ajustar los hiperparámetros
        grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_vec, y)

        # Obtener el mejor modelo con los mejores parámetros
        self.model = grid_search.best_estimator_
        print("Mejores hiperparámetros:", grid_search.best_params_)

    def get_response(self, user_input):
        if not user_input.strip():
            return "Por favor, escribe algo."

        # Convertir la entrada del usuario
        X_input = self.vectorizer.transform([user_input.lower()])
        # Predecir la categoría con Naive Bayes
        category_index = self.model.predict(X_input)[0]
        # Obtener la categoría predicha
        category = self.label_encoder.inverse_transform([category_index])[0]

        # Buscar las respuestas asociadas con la categoría
        possible_answers = [self.answers[i] for i, cat in enumerate(self.categories) if cat == category]
        if possible_answers:
            return np.random.choice(possible_answers)  # Selecciona una respuesta al azar para mayor dinamismo

        return "Lo siento, no encontré una respuesta adecuada."


# Añadir el saludo inicial y personalización
def iniciar_chat(chat_model):
    print("\u00a1Hola! Bienvenido al ChatBot.")
    nombre_usuario = input("Primero, dime tu nombre: ").strip()
    print(f"\u00a1Encantado de conocerte, {nombre_usuario}! Escribe 'fin' para terminar la conversación.")
    
    while True:
        user_input = input(f"{nombre_usuario}: ")
        if user_input.lower() == "fin":
            print(f"ChatBot: Hasta pronto, {nombre_usuario}.")
            break
        response = chat_model.get_response(user_input)
        print(f"ChatBot: {response}")


# Crear instancia del modelo y cargar datos
data_path = "data.json"
try:
    chat_model = ChatModel(data_path)
    iniciar_chat(chat_model)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {data_path}. Asegúrate de que exista y sea válido.")
