from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

def build_recommendation_model(recipes_df):
    """
    Construye un modelo de vecinos más cercanos utilizando la representación numérica
    de los ingredientes de cada receta.
    
    Args:
        recipes_df (DataFrame): DataFrame con las recetas seguras.
    
    Returns:
        model: Modelo NearestNeighbors entrenado.
        vectorizer: Objeto CountVectorizer utilizado para transformar el texto.
        ingredient_matrix: Matriz resultante de la transformación.
    """
    vectorizer = CountVectorizer(stop_words='english')
    ingredient_matrix = vectorizer.fit_transform(recipes_df['Cleaned_Ingredients'])
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(ingredient_matrix)
    return model, vectorizer, ingredient_matrix

def recommend_recipes(user_ingredients, model, vectorizer, recipes_df, n_recommendations=5):
    """
    Recomienda recetas basándose en los ingredientes que el usuario posee.
    
    Args:
        user_ingredients (list): Lista de ingredientes disponibles del usuario.
        model: Modelo NearestNeighbors entrenado.
        vectorizer: Objeto CountVectorizer utilizado en el modelo.
        recipes_df (DataFrame): DataFrame con las recetas seguras.
        n_recommendations (int): Número de recetas a recomendar.
    
    Returns:
        DataFrame: Recetas recomendadas.
    """
    query = " ".join(user_ingredients)
    user_vector = vectorizer.transform([query])
    distances, indices = model.kneighbors(user_vector, n_neighbors=n_recommendations)
    recommended = recipes_df.iloc[indices[0]]
    return recommended
