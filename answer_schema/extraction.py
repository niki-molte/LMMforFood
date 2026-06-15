from pydantic import BaseModel, Field

class IngredientExtraction(BaseModel):
    visible_ingredients: list[str] = Field(description="List of ingredients clearly visible in the dish.")
    hidden_ingredients: list[str] = Field(
        description="List of deduced hidden ingredients like spices, flour, eggs ,salt or oils.")
    allergens: list[str] = Field(
        description="List of potential allergens based on all ingredients like dairy, milk, gluten.")

class RecipeExtraction(BaseModel):
    recipe_name: str = Field(description="The guessed name of the recipe.")
    ingredients_with_quantities: list[str] = Field(
        description="List of ingredients with their estimated quantities (e.g., '200g pasta', '2 eggs').")
    steps: list[str] = Field(description="Chronological list of steps to prepare the recipe.")

class Macronutrients(BaseModel):
    calories: float = Field(
        description="Estimated energy in calories (kcal)."
    )
    fat: float = Field(
        description="Estimated fat in grams."
    )
    carbohydrates: float = Field(
        description="Estimated carbohydrates in grams."
    )
    protein: float = Field(
        description="Estimated protein in grams."
    )

class IngredientWithNutrition(BaseModel):
    name: str = Field(
        description="Name of the ingredient (e.g., 'tomato', 'olive oil', 'flour')."
    )
    weight_grams: float = Field(
        description="Estimated weight of this specific ingredient in grams."
    )
    macronutrients: Macronutrients = Field(
        description="Estimated macronutrients for this specific ingredient based on its weight."
    )

class NutritionExtraction(BaseModel):
    visible_ingredients: list[IngredientWithNutrition] = Field(
        description="List of ingredients clearly visible in the dish, along with their estimated weight and macronutrients."
    )
    hidden_ingredients: list[IngredientWithNutrition] = Field(
        description="List of deduced hidden ingredients like spices, flour, butter, salt, or oils, along with their estimated weight and macronutrients."
    )
    allergens: list[str] = Field(
        description="List of potential allergens based on all ingredients like dairy, milk, nuts, gluten."
    )
    total_macronutrients: Macronutrients = Field(
        description="The total estimated macronutrients and calories for the entire dish, calculated by summing up all visible and hidden ingredients."
    )