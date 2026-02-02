# Informe de tecnologías utilizadas

## Backend
- **Python 3 + Flask**: servidor web y ruteo de la aplicación, además de servir el frontend y el endpoint `/chat`.
- **scikit-learn**: vectorización de texto con `CountVectorizer` y clasificación con `MultinomialNB`.
- **NLTK**: limpieza lingüística en español mediante stopwords para mejorar la calidad del modelo.
- **NumPy**: selección aleatoria de respuestas dentro de la categoría predicha.

## Frontend
- **HTML5 + CSS3**: estructura y estilos de la interfaz del chatbot.
- **Bootstrap 5**: utilidades de diseño responsivo y componentes base.
- **JavaScript (Fetch API)**: envío de mensajes al backend y actualización dinámica de la conversación.

## Datos
- **JSON**: archivo `data.json` con el conjunto de preguntas, respuestas y categorías.
