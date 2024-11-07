import pandas as pd  # type: ignore
import joblib
from flask import Flask, render_template, request, redirect, url_for
from threading import Thread
import pyttsx3

app = Flask(__name__)

# Load the trained Gradient Boosting model
model = joblib.load('gradient_boosting_model.pkl')

# Load the LabelEncoders for company name, job title, and education
le_company = joblib.load('le_company.pkl')
le_job = joblib.load('le_job.pkl')
le_education = joblib.load('le_education.pkl')

# Function to convert salary numbers to words (Indian numbering system)
def convert_to_indian_words(num):
    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if num == 0:
        return "zero"

    words = ""
    if num >= 10000000:  # crore
        crore = num // 10000000
        words += convert_to_indian_words(crore) + " crore "
        num %= 10000000
    if num >= 100000:  # lakh
        lakh = num // 100000
        words += convert_to_indian_words(lakh) + " lakh "
        num %= 100000
    if num >= 1000:  # thousand
        thousand = num // 1000
        words += convert_to_indian_words(thousand) + " thousand "
        num %= 1000
    if num >= 100:  # hundreds
        hundred = num // 100
        words += units[hundred] + " hundred "
        num %= 100
    if num >= 20:  # tens
        ten = num // 10
        words += tens[ten] + " "
        num %= 10
    if num >= 11 and num <= 19:  # teens
        words += teens[num - 10] + " "
        num = 0
    if num > 0:  # units
        words += units[num] + " "

    return words.strip()

# Function to speak the predicted salary
def speak_salary(salary_text):
    engine = pyttsx3.init()
    engine.say(salary_text)
    engine.runAndWait()

# Main route to handle form submission and prediction
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data and handle case when experience is 'fresher' (0 years)
            company_name = request.form.get('company_name')
            job_title = request.form.get('job_title')
            min_experience = request.form.get('min_experience') 
            education_level = request.form.get('education_level')

            # Handle invalid or missing data
            if not company_name or not job_title or not education_level:
                return redirect(url_for('index'))
            
            # Handle fresher case explicitly, set min_experience to 0 if 'fresher'
            if min_experience == 'fresher':
                min_experience = 0
            else:
                min_experience = int(min_experience)

            # Encode the categorical features
            company_encoded = le_company.transform([company_name])[0]
            job_encoded = le_job.transform([job_title])[0]
            education_encoded = le_education.transform([education_level])[0]

            # Create a DataFrame with the appropriate feature names
            input_features = pd.DataFrame({
                'company_name': [company_encoded],
                'job_title': [job_encoded],
                'min_experience': [min_experience],
                'education_level': [education_encoded]
            })

            # Make the salary prediction
            predicted_salary = model.predict(input_features)[0]
            predicted_salary = round(predicted_salary)  # Round to the nearest whole number
            formatted_salary = "{:,.0f}".format(predicted_salary)  # Format with commas

            # Convert salary to Indian words for the voice assistant
            salary_in_words = convert_to_indian_words(predicted_salary)
            salary_text = f"The predicted salary is {salary_in_words} rupees."

            # Run the text-to-speech in a separate thread
            Thread(target=speak_salary, args=(salary_text,)).start()

            # Render the same page with the predicted salary
            return render_template('salary.html', predicted_salary=f"₹{formatted_salary} INR")

        except Exception as e:
            print(f"Error: {e}")
            return "An error occurred during the prediction.", 400

    # On GET request, show the form without a predicted salary
    return render_template('salary.html', predicted_salary=None)

if __name__ == '__main__':
    app.run(port=5000)
