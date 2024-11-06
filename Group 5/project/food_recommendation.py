import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and prepare the datasets
user_df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Smaartfir final\main\final_dataset_BFP.csv") # Ensure correct path
diet_df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Smaartfir final\main\diet_1.csv")
# Ensure correct path

# Encode categorical features
user_df['BFPcase'] = user_df['BFPcase'].str.lower()
user_df['Gender'] = user_df['Gender'].str.lower()
user_df['BMIcase'] = user_df['BMIcase'].str.lower()

le_bfp = LabelEncoder()
le_bfp.fit(user_df['BFPcase'])

le_gender = LabelEncoder()
le_gender.fit(user_df['Gender'])

le_bmi = LabelEncoder()
le_bmi.fit(user_df['BMIcase'])

user_df['BFPcase'] = le_bfp.transform(user_df['BFPcase'])
user_df['Gender'] = le_gender.transform(user_df['Gender'])
user_df['BMIcase'] = le_bmi.transform(user_df['BMIcase'])

# Define features and target
X = user_df[['Weight', 'Height', 'BMI', 'Gender', 'Age']]
y = user_df[['BFPcase', 'BMIcase']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=500, random_state=42)
clf.fit(X_train, y_train)

# Function to predict BMI and BFP
def predict_bmi_bfp(weight, height, age, gender):
    gender_encoded = le_gender.transform([gender.lower()])[0]
    bmi = weight / (height ** 2)

    user_input = pd.DataFrame({
        'Weight': [weight],
        'Height': [height],
        'BMI': [bmi],
        'Gender': [gender_encoded],
        'Age': [age]
    })

    prediction = clf.predict(user_input)
    predicted_bmi = le_bmi.inverse_transform([prediction[0][1]])[0]
    return predicted_bmi

# Nutrient targets based on BMI and fitness goal
targets = {
    'normal': {
        'weight loss': {'Calories': (1400, 1800), 'Protein': (60, 100)},
        'maintenance': {'Calories': (1800, 2200), 'Protein': (55, 90)},
        'muscle gain': {'Calories': (2200, 2800), 'Protein': (80, 130)}
    },
    'mild thinness': {
        'weight loss': {'Calories': (1500, 1800), 'Protein': (60, 100)},
        'maintenance': {'Calories': (2000, 2300), 'Protein': (70, 110)},
        'muscle gain': {'Calories': (2500, 2900), 'Protein': (100, 140)}
    },
    'moderate thinness': {
        'weight loss': {'Calories': (1600, 1900), 'Protein': (65, 105)},
        'maintenance': {'Calories': (2100, 2500), 'Protein': (80, 115)},
        'muscle gain': {'Calories': (2600, 3000), 'Protein': (110, 150)}
    },
    'sever thinness': {
        'weight loss': {'Calories': (1800, 2100), 'Protein': (70, 110)},
        'maintenance': {'Calories': (2300, 2700), 'Protein': (90, 125)},
        'muscle gain': {'Calories': (2800, 3200), 'Protein': (120, 160)}
    },
    'overweight': {
        'weight loss': {'Calories': (1200, 1500), 'Protein': (55, 90)},
        'maintenance': {'Calories': (1700, 2000), 'Protein': (60, 100)},
        'muscle gain': {'Calories': (2200, 2600), 'Protein': (85, 125)}
    },
    'obese': {
        'weight loss': {'Calories': (1000, 1300), 'Protein': (60, 90)},
        'maintenance': {'Calories': (1400, 1700), 'Protein': (50, 80)},
        'muscle gain': {'Calories': (1800, 2200), 'Protein': (75, 115)}
    },
    'sever obese': {
        'weight loss': {'Calories': (900, 1100), 'Protein': (50, 80)},
        'maintenance': {'Calories': (1200, 1500), 'Protein': (55, 85)},
        'muscle gain': {'Calories': (1600, 2000), 'Protein': (70, 100)}
    }
}

# Function to evaluate the fitness of a meal plan
def evaluate_fitness(meal_plan, calorie_target, protein_target):
    total_calories = meal_plan['Calories (kcal)'].sum()
    total_protein = meal_plan['Protein (g)'].sum()
    calorie_diff = abs(total_calories - calorie_target)
    protein_diff = abs(total_protein - protein_target)
    return 1 / (1 + calorie_diff + protein_diff)  # Higher is better

# Function to generate a random meal plan
def generate_random_meal_plan(diet_df, meal_type, excluded_dishes, max_dishes=3):
    meal_df = diet_df[diet_df['MealType'].str.lower() == meal_type.lower()]
    meal_df = meal_df[~meal_df['Dish'].isin(excluded_dishes)]
    if meal_df.empty:
        return pd.DataFrame()

    num_dishes = len(meal_df)
    num_selected = min(np.random.randint(1, max_dishes + 1), num_dishes)
    selected_dishes = meal_df.sample(n=num_selected)
    return selected_dishes

# Genetic Algorithm for meal planning
def genetic_algorithm(diet_df, calorie_target, protein_target, meal_type, excluded_dishes, num_generations=20, population_size=20, mutation_rate=0.2, max_dishes=3):
    def create_population():
        population = []
        for _ in range(population_size):
            meal_plan = generate_random_meal_plan(diet_df, meal_type, excluded_dishes, max_dishes)
            population.append(meal_plan)
        return population

    def crossover(parent1, parent2):
        combined = pd.concat([parent1, parent2]).drop_duplicates()
        return combined.sample(n=min(len(combined), max_dishes))

    def mutate(meal_plan):
        meal_df = diet_df[diet_df['MealType'].str.lower() == meal_type.lower()]
        num_dishes = len(meal_df)
        if num_dishes > 0:
            mutation_prob = random.random()
            if mutation_prob < mutation_rate:
                new_dish = meal_df.sample(n=1)
                meal_plan = pd.concat([meal_plan, new_dish]).drop_duplicates()
                if len(meal_plan) > max_dishes:
                    meal_plan = meal_plan.sample(n=max_dishes)
        return meal_plan

    def select(population):
        scores = [evaluate_fitness(plan, calorie_target, protein_target) for plan in population]
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores]
        selected_indices = np.random.choice(range(len(population)), size=2, p=probabilities)
        return [population[i] for i in selected_indices]

    population = create_population()
    best_plan = None
    best_fitness = 0

    for generation in range(num_generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select(population)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.extend([child1, child2])

        population = new_population

        for plan in population:
            fitness = evaluate_fitness(plan, calorie_target, protein_target)
            if fitness > best_fitness:
                best_fitness = fitness
                best_plan = plan

    return best_plan

# Function to generate the meal plan based on the predicted BMI and user fitness goal
def get_meal_plan(predicted_bmi, fitness_goal, weight):
    if predicted_bmi in targets and fitness_goal in targets[predicted_bmi]:
        calorie_range = targets[predicted_bmi][fitness_goal]['Calories']
        protein_range = targets[predicted_bmi][fitness_goal]['Protein']

        # Calculate per meal targets
        total_calories = np.random.randint(*calorie_range)
        total_protein = np.random.randint(*protein_range)
        per_meal_calories = total_calories // 4
        per_meal_protein = total_protein // 4

        meal_plan = {}

        for meal_type in ['Breakfast', 'Lunch', 'Dinner', 'Snack']:
            excluded_dishes = set()
            plan = genetic_algorithm(diet_df, per_meal_calories, per_meal_protein, meal_type, excluded_dishes)
            meal_plan[meal_type] = plan if plan is not None else pd.DataFrame()

        return meal_plan
    else:
        print("Invalid BMI or fitness goal.")
        return None
