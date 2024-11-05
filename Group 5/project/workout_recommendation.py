import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
import csv

# Load the dataset and preprocess
def load_user_data():
    user_df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Smaartfir final\main\final_dataset_BFP.csv")
    user_df['BFPcase'] = user_df['BFPcase'].str.lower()
    user_df['Gender'] = user_df['Gender'].str.lower()
    user_df['BMIcase'] = user_df['BMIcase'].str.lower()

    le_bfp = LabelEncoder()
    le_gender = LabelEncoder()
    le_bmi = LabelEncoder()

    user_df['BFPcase'] = le_bfp.fit_transform(user_df['BFPcase'])
    user_df['Gender'] = le_gender.fit_transform(user_df['Gender'])
    user_df['BMIcase'] = le_bmi.fit_transform(user_df['BMIcase'])

    return user_df, le_bfp, le_gender, le_bmi

# Predict BFP and BMI cases based on user input
def predict_cases(weight, height, gender, age):
    user_df, le_bfp, le_gender, le_bmi = load_user_data()
    X = user_df[['Weight', 'Height', 'BMI', 'Gender', 'Age']]
    y = user_df[['BFPcase', 'BMIcase']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=70)
    
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=90, random_state=30))
    clf.fit(X_train, y_train)
    
    bmi = weight / (height ** 2)
    gender_encoded = le_gender.transform([gender])[0]
    user_data = pd.DataFrame({
        'Weight': [weight],
        'Height': [height],
        'BMI': [bmi],
        'Gender': [gender_encoded],
        'Age': [age]
    })

    prediction = clf.predict(user_data)
    predicted_bfpcase = le_bfp.inverse_transform([prediction[0][0]])[0]
    predicted_bmi = le_bmi.inverse_transform([prediction[0][1]])[0]

    return predicted_bfpcase, predicted_bmi

# Load workout data from CSV
def load_workout_data():
    workout_data = []
    with open(r"C:\Users\HP\OneDrive\Desktop\Smaartfir final\main\with gifs csv.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            exercise_data = {
                "exercise": row["Exercise Name"],
                "muscle_group": row["Muscle Group"],
                "intensity": row["Intensity Level"],
                "bmi_case": [bmi.strip() for bmi in row["Suitable for BMI Case"].split(',')],
                "bfp_case": [bfp.strip() for bfp in row["Suitable for BFP Case"].split(',')],
                "reps": row["Reps"],
                "sets": row["Sets"],
                "duration": row["Duration (minutes)"],
                "rest": row["Rest (seconds)"],
                "gifs": row["Gifs"],
                "instructions": row["instructions"]

            }
            workout_data.append(exercise_data)
    return workout_data

# Weekly workout plan specification
weekly_plan_spec = {
    'Day 1': ['chest', 'triceps'],
    'Day 2': ['back', 'biceps'],
    'Day 3': ['shoulder', 'legs'],
    'Day 4': ['abs', 'triceps'],
    'Day 5': ['back', 'shoulder'],
    'Day 6': ['legs', 'chest'],
    'Day 7': ['rest']
}

# Fitness function to evaluate a workout plan
def fitness_function(plan, user_bmi_case, user_bfp_case):
    fitness = 0
    used_muscles = set()
    for day, exercises in plan.items():
        for exercise in exercises:
            if user_bmi_case in exercise['bmi_case'] and user_bfp_case in exercise['bfp_case']:
                fitness += 1
            used_muscles.add(exercise['muscle_group'])
    for day, muscle_groups in weekly_plan_spec.items():
        day_muscles = {exercise['muscle_group'] for exercise in plan[day] if exercise['muscle_group'] != 'rest'}
        if day_muscles != set(muscle_groups):
            fitness -= 1
    return fitness

# Generate a random individual (workout plan)
def generate_individual(workout_data):
    plan = {}
    exercises_per_muscle_group = 3
    for day, muscle_groups in weekly_plan_spec.items():
        plan[day] = []
        for muscle_group in muscle_groups:
            if muscle_group != 'rest':
                suitable_exercises = [ex for ex in workout_data if ex['muscle_group'].lower() == muscle_group]
                if len(suitable_exercises) >= exercises_per_muscle_group:
                    selected_exercises = random.sample(suitable_exercises, exercises_per_muscle_group)
                else:
                    selected_exercises = suitable_exercises
                plan[day].extend(selected_exercises)
        if len(plan[day]) < 6 and muscle_groups != ['rest']:
            remaining_exercises = 6 - len(plan[day])
            additional_exercises = random.sample(workout_data, remaining_exercises)
            plan[day].extend(additional_exercises[:remaining_exercises])
    return plan

# Generate a population of workout plans
def generate_population(workout_data, size):
    return [generate_individual(workout_data) for _ in range(size)]

# Select two parents for crossover
def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [fitness / total_fitness for fitness in fitness_scores]
    parent1, parent2 = random.choices(population, weights=selection_probs, k=2)
    return parent1, parent2

# Crossover function to combine two parents
def crossover(parent1, parent2):
    crossover_point = random.choice(list(weekly_plan_spec.keys()))
    child1, child2 = {}, {}
    for day in weekly_plan_spec:
        if day <= crossover_point:
            child1[day] = parent1[day]
            child2[day] = parent2[day]
        else:
            child1[day] = parent2[day]
            child2[day] = parent1[day]
    return child1, child2

# Mutate a workout plan
def mutate(individual, workout_data, mutation_rate=0.1):
    for day in individual:
        if random.random() < mutation_rate:
            for i, exercise in enumerate(individual[day]):
                suitable_exercises = [ex for ex in workout_data if ex['muscle_group'].lower() == exercise['muscle_group']]
                if suitable_exercises:
                    individual[day][i] = random.choice(suitable_exercises)
    return individual

# Genetic algorithm to evolve workout plans
def genetic_algorithm(workout_data, user_bmi_case, user_bfp_case, population_size=20, generations=50, mutation_rate=0.1):
    population = generate_population(workout_data, population_size)
    for generation in range(generations):
        fitness_scores = [fitness_function(ind, user_bmi_case, user_bfp_case) for ind in population]
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, workout_data, mutation_rate)
            child2 = mutate(child2, workout_data, mutation_rate)
            new_population.extend([child1, child2])
        population = new_population
    final_fitness_scores = [fitness_function(ind, user_bmi_case, user_bfp_case) for ind in population]
    best_individual = population[final_fitness_scores.index(max(final_fitness_scores))]
    return best_individual
