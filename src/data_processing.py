import pandas as pd
import ast

def load_food_data(food_data_path):
    """
    Carga el dataset de alimentos y alergias.
    
    Args:
        food_data_path (str): Ruta al archivo CSV con datos de alimentos y alergias.
    
    Returns:
        DataFrame: Datos de alimentos.
    """
    return pd.read_csv(food_data_path)

def load_recipes(recipes_path):
    """
    Carga el dataset de recetas.
    
    Args:
        recipes_path (str): Ruta al archivo CSV con datos de recetas.
    
    Returns:
        DataFrame: Datos de recetas.
    """
    recipes = pd.read_csv(recipes_path)
    recipes['Cleaned_Ingredients'] = recipes['Cleaned_Ingredients'].apply(process_ingredients)
    return recipes

def create_allergy_mapping(food_data):
    """
    Crea un diccionario que mapea cada alergia a la lista de alimentos que la provocan.
    
    Args:
        food_data (DataFrame): Datos de alimentos y alergias.
    
    Returns:
        dict: Mapeo de alergias a alimentos.
    """
    return food_data.groupby('Allergy')['Food'].apply(list).to_dict()

def process_ingredients(ingredients_str):
    """
    Convierte una cadena que representa una lista de ingredientes en una cadena
    concatenada para facilitar su vectorización.
    
    Args:
        ingredients_str (str): Cadena en formato de lista (por ejemplo, "['apple', 'chicken']").
    
    Returns:
        str: Cadena resultante con todos los ingredientes concatenados.
    """
    try:
        ingredients_list = ast.literal_eval(ingredients_str)
        return " ".join(ingredients_list)
    except Exception:
        return ingredients_str

def contains_allergen(ingredients_text, user_allergies, allergy_mapping):
    """
    Verifica si el texto de ingredientes contiene algún alimento relacionado
    con las alergias especificadas por el usuario.
    
    Args:
        ingredients_text (str): Texto con la lista de ingredientes.
        user_allergies (list): Lista de alergias del usuario (por ejemplo, ['Nut Allergy']).
        allergy_mapping (dict): Diccionario que mapea alergias a alimentos.
    
    Returns:
        bool: True si se encuentra al menos un alérgeno, False en caso contrario.
    """
    text = ingredients_text.lower()
    for allergy in user_allergies:
        if allergy in allergy_mapping:
            for food in allergy_mapping[allergy]:
                if food.lower() in text:
                    return True
    return False

def filter_recipes_by_allergies(recipes_df, user_allergies, allergy_mapping):
    """
    Filtra las recetas eliminando aquellas que contengan ingredientes asociados a
    las alergias declaradas por el usuario.
    
    Args:
        recipes_df (DataFrame): DataFrame con las recetas.
        user_allergies (list): Lista de alergias del usuario.
        allergy_mapping (dict): Diccionario que mapea alergias a alimentos.
    
    Returns:
        DataFrame: Recetas que son seguras para el usuario.
    """
    filtered = recipes_df[~recipes_df['Cleaned_Ingredients'].apply(
        lambda x: contains_allergen(x, user_allergies, allergy_mapping)
    )]
    return filtered
