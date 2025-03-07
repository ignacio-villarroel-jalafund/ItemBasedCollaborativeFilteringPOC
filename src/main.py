from data_processing import (
    load_food_data, load_recipes, create_allergy_mapping,
    filter_recipes_by_allergies
)
from item_recommender import build_recommendation_model, recommend_recipes

def main():
    FOOD_DATA_PATH = './datasets/FoodData.csv'
    RECIPES_PATH = './datasets/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
    
    food_data = load_food_data(FOOD_DATA_PATH)
    recipes = load_recipes(RECIPES_PATH)
    
    allergy_mapping = create_allergy_mapping(food_data)
    
    user_allergies = ['Nut Allergy']
    
    user_ingredients = ['apple', 'chicken', 'rice']
    
    safe_recipes = filter_recipes_by_allergies(recipes, user_allergies, allergy_mapping)
    
    if safe_recipes.empty:
        print("No se encontraron recetas seguras que cumplan con los criterios de alergia del usuario.")
        return
    
    model, vectorizer, _ = build_recommendation_model(safe_recipes)
    
    recommendations = recommend_recipes(user_ingredients, model, vectorizer, safe_recipes, n_recommendations=5)
    
    print("Recetas recomendadas para el usuario:")
    print(recommendations[['Title', 'Cleaned_Ingredients']])
    
if __name__ == '__main__':
    main()
