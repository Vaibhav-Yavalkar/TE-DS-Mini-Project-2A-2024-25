from flask import Flask, render_template, request 
from food_recommendation import predict_bmi_bfp, get_meal_plan  # Import functions from app2.py
from workout_recommendation import predict_cases, load_workout_data, genetic_algorithm  # Import functions from app3.py
from flask import Flask, render_template, Response, request
from food_recommendation import predict_bmi_bfp, get_meal_plan  # For diet recommendation system
from workout_recommendation import predict_cases, load_workout_data, genetic_algorithm  # For workout recommendation system
from bicep_curl import bicep_curl_frames  # For OpenCV exercise tracking
from squat import squat_frames
from jumping_jacks import jumping_jacks_frames
from lunges import lunges_frames
import math
app = Flask(__name__)

# Route for the index page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/dashboard')
def index():
    return render_template('index.html')

@app.route('/calculator')
def cal():
    return render_template('cal.html')

@app.route('/workout_plan')
def workout_plan():
    return render_template('workoutpla.html')
@app.route('/gymlogger')
def gymlogger():
    return render_template('gymlogger.html')



# =========================DIET RECOMMENDATION=======================
# Route for the diet recommendation system
@app.route('/diet')
def diet():
    return render_template('diet.html')

# Route to generate meal plan (app2.py)
@app.route('/generate_meal_plan', methods=['POST'])
def generate_meal_plan():
    # Get the form data
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    age = int(request.form['age'])
    gender = request.form['gender']
    fitness_goal = request.form['fitness_goal']

    # Predict the BMI
    predicted_bmi = predict_bmi_bfp(weight, height, age, gender)

    # Generate the meal plan
    meal_plan = get_meal_plan(predicted_bmi, fitness_goal, weight)

    # Prepare the meal plan for display (convert to a readable format)
    meal_plan_summary = {}
    for meal_type, plan in meal_plan.items():
        if not plan.empty:
            meal_plan_summary[meal_type] = plan[['Dish', 'Calories (kcal)', 'Protein (g)','Ingredients','Recipe']].to_dict(orient='records')
        else:
            meal_plan_summary[meal_type] = []

    # Pass the meal plan to result.html
    return render_template('food_recommendation.html', meal_plan=meal_plan_summary)

# =========================DIET RECOMMENDATION END =======================


# Route for the workout recommendation system (app3.py)
@app.route('/workout')
def workout():
    return render_template('workout.html')

# Route to generate workout plan (app3.py)
@app.route('/generate_workout', methods=['POST'])
def generate_workout():
    # Get the form data
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    age = int(request.form['age'])
    gender = request.form['gender'].lower()

    # Predict the BFP and BMI cases using app3.py
    predicted_bfpcase, predicted_bmi = predict_cases(weight, height, gender, age)

    # Load workout data and generate the workout plan using genetic algorithm
    workout_data = load_workout_data()
    best_plan = genetic_algorithm(workout_data, predicted_bmi, predicted_bfpcase)

    # Pass the workout plan to result3.html
    return render_template('workout_result.html',workout_plan=best_plan)

# =========================DIET RECOMMENDATION END =======================

# =======================open cv ==============================
@app.route('/exercise')
def exercise():
    return render_template('opencv.html')  # Page to choose exercises

@app.route('/bicep_curl')
def bicep_curl():
    return Response(bicep_curl_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/squat')
def squat():
    return Response(squat_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/jumping_jacks')
def jumping_jacks():
    return Response(jumping_jacks_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/lunges')
def lunges():
    return Response(lunges_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# ===============================open cv =====================

#=================BMI Calculator=============================#
@app.route('/bmi', methods=['GET', 'POST'])
def bmi_calculator():
    bmi = None
    height = None
    weight = None
    age = None
    message = None

    if request.method == 'POST':
        try:
            height = float(request.form['userheight'])
            weight = float(request.form['userweight'])
            age = int(request.form['userage'])
            height_in_meters = height / 100
            bmi = round(weight / (height_in_meters ** 2), 2)

            # Provide additional messages based on age and BMI
            if age < 18:
                message = "BMI results for children and teens vary by age and gender."
            elif 18 <= age <= 65:
                if bmi < 18.5:
                    message = "Underweight"
                elif 18.5 <= bmi < 24.9:
                    message = "Normal weight"
                elif 25 <= bmi < 29.9:
                    message = "Overweight"
                else:
                    message = "Obese"
            else:
                message = "Consult a healthcare provider for seniors' BMI interpretation."

        except (ValueError, ZeroDivisionError):
            bmi = 'Invalid input, please enter valid numbers.'

    return render_template('bmi_calculator.html', height=height, weight=weight, bmi=bmi, age=age, message=message)

#=================BMI Calculator=============================#

#===========================Calorie Intake Calculator=======================#
def calculate_bmr(weight, height, age, gender):
    if gender == 'male':
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    return bmr

def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9,
    }
    return bmr * activity_multipliers.get(activity_level, 1)

def calculate_goal_calories(tdee, goal):
    goal_multipliers = {
        'extreme_loss': 0.7,
        'moderate_loss': 0.8,
        'maintenance': 1.0,
        'moderate_gain': 1.2,
        'extreme_gain': 1.3,
    }
    return tdee * goal_multipliers.get(goal, 1)

@app.route('/calorie', methods=['GET', 'POST'])
def calorie_calculator():
    total_calories = None
    weight = None
    height = None
    age = None
    gender = None
    activity_level = None
    goal = None

    if request.method == 'POST':
        try:
            weight = float(request.form.get('weight', 0))
            height = float(request.form.get('height', 0))
            age = int(request.form.get('age', 0))
            gender = request.form.get('gender')
            activity_level = request.form.get('activity_level')
            goal = request.form.get('goal')

            if weight > 0 and height > 0 and age > 0 and gender and activity_level and goal:
                bmr = calculate_bmr(weight, height, age, gender)
                tdee = calculate_tdee(bmr, activity_level)
                total_calories = calculate_goal_calories(tdee, goal)
            else:
                total_calories = 'Please enter all required fields.'
        except ValueError:
            total_calories = 'Invalid input, please enter valid numbers.'

    return render_template('calorie.html', total_calories=total_calories)
#===========================Calorie Intake Calculator=======================#

#====================Ideal Weight Calculator=======================#
def calculate_ideal_weight(height, age, gender):
    if gender == 'male':
        ideal_weight = (height - 100) + (age / 10)
    else:
        ideal_weight = (height - 100) + (age / 20)
    return ideal_weight

@app.route('/ideal', methods=['GET', 'POST'])
def weight_ideal():
    ideal_weight = None
    if request.method == 'POST':
        height = float(request.form['height'])
        age = int(request.form['age'])
        gender = request.form['gender']
        ideal_weight = calculate_ideal_weight(height, age, gender)
    return render_template('ideal.html', ideal_weight=ideal_weight)
#====================Ideal Weight Calculator=======================#

#====================Body Fat Percentage===========================#
def calculate_body_fat_percentage(gender, weight, waist, neck, height):
    if gender == 'male':
        body_fat = 86.010 * math.log10(waist - neck) - 70.041 * math.log10(height) + 36.76
    else:
        body_fat = 163.205 * math.log10(waist - neck) - 97.684 * math.log10(height) - 78.387
    return round(body_fat, 2)

@app.route('/body_fatt', methods=['GET', 'POST'])
def body_fat_percentage():
    body_fat = None
    if request.method == 'POST':
        try:
            gender = request.form['gender']
            weight = float(request.form['weight'])
            waist = float(request.form['waist'])
            neck = float(request.form['neck'])
            height = float(request.form['height'])
            body_fat = calculate_body_fat_percentage(gender, weight, waist, neck, height)
        except ValueError:
            body_fat = 'Invalid input, please enter valid numbers.'
    return render_template('body_fat.html', body_fat=body_fat)

#====================Body Fat Percentage===========================#

if __name__ == '__main__':
    app.run(debug=True)
