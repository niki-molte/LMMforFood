from pydantic import BaseModel, Field


class IngredientExtraction(BaseModel):
    visible_ingredients: list[str] = Field(description="List of ingredients clearly visible in the dish.")
    hidden_ingredients: list[str] = Field(description="List of deduced hidden ingredients like spices, flour, eggs ,salt or oils.")
    allergens: list[str] = Field(description="List of potential allergens based on all ingredients like dairy, milk, gluten.")


class RecipeExtraction(BaseModel):
    recipe_name: str = Field(description="The guessed name of the recipe.")
    ingredients_with_quantities: list[str] = Field(
        description="List of ingredients with their estimated quantities (e.g., '200g pasta', '2 eggs').")
    steps: list[str] = Field(description="Chronological list of steps to prepare the recipe.")
