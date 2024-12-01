import base64
import os
import pickle
from flask import Flask, Response, render_template, request, redirect, url_for, session, send_file, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import mysql.connector
from mysql.connector import Error
from io import BytesIO
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import os
import requests 
from flask import render_template, url_for

app = Flask(__name__)
app.secret_key = 'your_secure_random_secret_key_here'  # Replace with your secure key
model = joblib.load('expense_prediction_model.pkl')
poly = joblib.load('poly_transform.pkl')

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Parthavi@1204',
    'database': 'digifnance'
}

def create_connection():
    """Create a database connection."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        if connection.is_connected():
            print("Connection to MySQL DB successful")
        else:
            print("Failed to connect to the database.")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    
    cursor.execute("SELECT SUM(amount) AS total_balance FROM income_expense WHERE user_id = %s AND type='income'", (user_id,))
    income_result = cursor.fetchone()
    total_income = income_result['total_balance'] or 0

    cursor.execute("SELECT SUM(amount) AS total_expense FROM income_expense WHERE user_id = %s AND type='expense'", (user_id,))
    expense_result = cursor.fetchone()
    total_expense = expense_result['total_expense'] or 0

    balance = total_income - total_expense

    cursor.execute("""
        SELECT category, SUM(amount) AS total_amount
        FROM income_expense
        WHERE user_id = %s AND type='expense'
        GROUP BY category
    """, (user_id,))
    expenses_by_category = cursor.fetchall()

    cursor.execute("""
        SELECT purpose, category, amount, date
        FROM income_expense
        WHERE user_id = %s
        ORDER BY date DESC
        LIMIT 5
    """, (user_id,))
    recent_transactions = cursor.fetchall()

    cursor.close()
    connection.close()

    return render_template('index.html', balance=balance, expenses_by_category=expenses_by_category, recent_transactions=recent_transactions)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        connection = create_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()

        cursor.close()
        connection.close()

        if user and check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['user_id'] = user['id']
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    connection = create_connection()
    cursor = connection.cursor()

    try:
        cursor.execute('INSERT INTO users (username, email, password) VALUES (%s, %s, %s)',
                       (username, email, hashed_password))
        connection.commit()
        return redirect(url_for('login'))
    except Error as e:
        print(f"The error '{e}' occurred")
        return render_template('login.html', error='Registration failed. Email may already be in use.')
    finally:
        cursor.close()
        connection.close()

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/delete_transaction/<int:transaction_id>', methods=['POST'])
def delete_transaction(transaction_id):
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = create_connection()
    cursor = connection.cursor()

    try:
        query = "DELETE FROM income_expense WHERE id = %s AND user_id = %s"
        cursor.execute(query, (transaction_id, user_id))
        connection.commit()
        print(f"Transaction with ID {transaction_id} deleted.")
    except Error as e:
        print(f"The error '{e}' occurred")
    finally:
        cursor.close()
        connection.close()

    return redirect(url_for('incexp'))

@app.route('/incexp',methods=['POST'])
def incexp():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM income_expense WHERE user_id = %s", (user_id,))
    transactions = cursor.fetchall()
    cursor.close()
    connection.close()

    # Generate and save the chart
    img = BytesIO()
    categories = {}
    for transaction in transactions:
        if transaction['type'] == 'expense':
            if transaction['category'] not in categories:
                categories[transaction['category']] = 0
            categories[transaction['category']] += transaction['amount']
    
    labels = list(categories.keys())
    data = list(categories.values())

    plt.figure(figsize=(8, 6))
    plt.bar(labels, data, color='skyblue')
    plt.xlabel('Category')
    plt.ylabel('Amount')
    plt.title('Expenses by Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)

    return render_template('incexp.html', transactions=transactions, chart_url=url_for('chart'))
@app.route('/chart')
def chart():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    chart_type = request.args.get('chart_type', 'category-bar')
    user_id = session['user_id']
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM income_expense WHERE user_id = %s", (user_id,))
    transactions = cursor.fetchall()
    cursor.close()
    connection.close()

    img = BytesIO()
    
    if chart_type == 'income':
        # Generate an Income Line Graph
        dates = [t['date'] for t in transactions if t['type'] == 'income']
        amounts = [t['amount'] for t in transactions if t['type'] == 'income']
        plt.figure(figsize=(8, 6))
        plt.plot(dates, amounts, marker='o', color='green')
        plt.xlabel('Date')
        plt.ylabel('Amount')
        plt.title('Income Over Time')
    elif chart_type == 'expense':
        # Generate an Expense Line Graph
        dates = [t['date'] for t in transactions if t['type'] == 'expense']
        amounts = [t['amount'] for t in transactions if t['type'] == 'expense']
        plt.figure(figsize=(8, 6))
        plt.plot(dates, amounts, marker='o', color='red')
        plt.xlabel('Date')
        plt.ylabel('Amount')
        plt.title('Expenses Over Time')
    elif chart_type == 'overall':
        dates = [t['date'] for t in transactions]
        amounts = [t['amount'] for t in transactions]
        colors = ['green' if t['type'] == 'income' else 'red' for t in transactions]
        plt.figure(figsize=(8, 6))
        # Sort transactions by date before plotting
        sorted_transactions = sorted(zip(dates, amounts, colors), key=lambda x: x[0])
        sorted_dates, sorted_amounts, sorted_colors = zip(*sorted_transactions)
        plt.plot(sorted_dates, sorted_amounts, color='blue', marker='o')  # Line graph
        plt.scatter(sorted_dates, sorted_amounts, c=sorted_colors)  # Scatter to show color distinction
        plt.xlabel('Date')
        plt.ylabel('Amount')
        plt.title('Overall Transactions Over Time')
    else:
        # Generate the Category Bar Chart (default)
        categories = {}
        for transaction in transactions:
            if transaction['type'] == 'expense':
                if transaction['category'] not in categories:
                    categories[transaction['category']] = 0
                categories[transaction['category']] += transaction['amount']
        
        labels = list(categories.keys())
        data = list(categories.values())

        plt.figure(figsize=(8, 6))
        plt.bar(labels, data, color='skyblue')
        plt.xlabel('Category')
        plt.ylabel('Amount')
        plt.title('Expenses by Category')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)

    return send_file(img, mimetype='image/png')


@app.route('/add_income_expense', methods=['POST'])
def add_income_expense():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    user_id = session['user_id']
    amount = float(request.form.get('amount'))  # Convert amount to float
    category = request.form.get('category')
    date = request.form.get('date')
    purpose = request.form.get('purpose')
    record_type = request.form.get('type')

    connection = create_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        # Insert the new transaction
        query = """
        INSERT INTO income_expense (user_id, amount, category, date, purpose, type)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (user_id, amount, category, date, purpose, record_type))
        connection.commit()

        # Only check budget if it's an expense
        if record_type == 'expense':
            # Fetch the budget for the category
            cursor.execute("""
                SELECT amount FROM budgets WHERE user_id = %s AND category = %s ORDER BY date DESC LIMIT 1
            """, (user_id, category))
            budget = cursor.fetchone()

            if budget:
                # Fetch the total spent amount for this category
                cursor.execute("""
                    SELECT SUM(amount) AS total_spent FROM income_expense
                    WHERE user_id = %s AND category = %s AND type = 'expense'
                """, (user_id, category))
                spent_data = cursor.fetchone()

                spent_amount = spent_data['total_spent'] if spent_data['total_spent'] else 0
                remaining_budget = budget['amount'] - spent_amount

                # Check if the budget is exceeded
                if remaining_budget < 0:
                    session['over_budget_warning'] = f"Warning: You have exceeded your budget for the '{category}' category!"

        return redirect('/')
    except Error as e:
        print(f"The error '{e}' occurred")
        return "An error occurred while adding the record."
    finally:
        cursor.close()
        connection.close()


@app.route('/budget')
def budget_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Render the budget.html page
    return render_template('budget.html')
@app.route('/set_budget', methods=['POST'])
def set_budget():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    amount = request.form.get('amount')
    category = request.form.get('category')
    date = request.form.get('date')
    user_id = session.get('user_id')

    if not amount or not amount.strip():
        return render_template('budget.html', error="Budget amount cannot be empty")

    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO budgets (user_id, amount, category, date)
        VALUES (%s, %s, %s, %s)
    """, (user_id, amount, category, date))
    connection.commit()
    cursor.close()
    connection.close()

    return redirect(url_for('budget_page'))

def calculate_remaining_budget(user_id, category, budget_amount, budget_date):
    connection = create_connection()
    cursor = connection.cursor()

    try:
        # Calculate the total expenses for the category after the budget date
        query_expenses = """
        SELECT SUM(amount) FROM income_expense 
        WHERE user_id = %s AND category = %s AND date >= %s
        """
        cursor.execute(query_expenses, (user_id, category, budget_date))
        total_expense = cursor.fetchone()[0]

        # Ensure budget_amount and total_expense are not None
        if budget_amount is None:
            print(f"Error: budget_amount is None for user_id: {user_id}, category: {category}")
            return None

        if total_expense is None:
            total_expense = 0

        # Calculate remaining amount
        remaining_amount = budget_amount - total_expense

        return remaining_amount

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

    finally:
        cursor.close()
        connection.close()

@app.route('/budget_statistics')
def budget_statistics():
    user_id = session['user_id']  # Assuming user_id is stored in session
    budget_stats = get_budget_statistics(user_id)  # Call the function to get the statistics

    return render_template('budget_statistics.html', budget_stats=budget_stats)
def get_budget_statistics(user_id):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        # Fetch all budgets for the user
        query_budgets = """
        SELECT id, category, amount, date FROM budgets 
        WHERE user_id = %s
        """
        cursor.execute(query_budgets, (user_id,))
        budgets = cursor.fetchall()

        budget_stats = []

        # Calculate remaining budget for each category and budget
        for budget in budgets:
            budget_id = budget['id']
            category = budget['category']
            budget_amount = budget['amount'] if budget['amount'] is not None else 0
            budget_date = budget['date']

            spent_amount = get_spent_amount(user_id, category, budget_date)

            if spent_amount is None:
                spent_amount = 0

            remaining_amount = budget_amount - spent_amount

            # Only append to budget_stats if the category and budget date are not None
            if category and budget_date:
                budget_stats.append({
                    "id": budget_id,
                    "Category": category,
                    "Budget Amount": budget_amount,
                    "Spent Amount": spent_amount,
                    "Budget Date": budget_date,
                    "Remaining Amount": remaining_amount,
                    "Graph": url_for('budget_graph', budget_id=budget_id)
                })

        return budget_stats

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []

    finally:
        cursor.close()
        connection.close()

def get_spent_amount(user_id, category, budget_date):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)  # Enable dictionary output

    try:
        query = """
        SELECT SUM(amount) as spent_amount FROM income_expense 
        WHERE user_id = %s AND category = %s AND date >= %s
        """
        cursor.execute(query, (user_id, category, budget_date))
        result = cursor.fetchone()

        if result is None:
            return 0  # Return 0 if no result is found

        return result['spent_amount']

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

    finally:
        cursor.close()
        connection.close()


@app.route('/budget_graph/<int:budget_id>')
def budget_graph(budget_id):
    user_id = session.get('user_id')  # Assuming the user ID is stored in the session
    if not user_id:
        abort(403, description="Unauthorized access")

    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Fetch budget data for the specific user
    cursor.execute("""
        SELECT category, amount, date
        FROM budgets
        WHERE id = %s AND user_id = %s
    """, (budget_id, user_id))
    budget_data = cursor.fetchone()

    if not budget_data:
        cursor.close()
        connection.close()
        abort(404, description="Budget data not found")

    # Fetch total spent amount for the category within the budget period
    cursor.execute("""
        SELECT SUM(amount) AS total_spent
        FROM income_expense
        WHERE category = %s AND date >= %s AND type = 'expense' AND user_id = %s
    """, (budget_data['category'], budget_data['date'], user_id))
    spent_data = cursor.fetchone()

    spent_amount = spent_data['total_spent'] if spent_data['total_spent'] is not None else 0
    
    cursor.close()
    connection.close()

    # Calculate remaining budget
    remaining_budget = budget_data['amount'] - spent_amount
    over_budget = spent_amount > budget_data['amount']

    # Generate graph
    fig, ax = plt.subplots(figsize=(10, 6))
    dates = pd.date_range(start=budget_data['date'], periods=1, freq='M')

    # Plot budget amount as a dashed line
    ax.plot(dates, [budget_data['amount']] * len(dates), label='Budget Amount', color='blue', linestyle='--')

    # Plot spent amount as a bar
    ax.bar(dates, [spent_amount] * len(dates), width=20, label='Spent Amount', color='blue', alpha=0.7)

    # Show alert message if over budget
    if over_budget:
        ax.text(dates[-1], budget_data['amount'], 'Over Budget!', color='red', fontsize=12, fontweight='bold', ha='center')

    ax.set_xlabel('Date')
    ax.set_ylabel('Amount')
    ax.set_title(f'Budget vs Spent Amount for Budget ID {budget_id}')
    ax.legend()
    ax.grid(True)

    # Convert plot to PNG image in memory
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)


    return send_file(io.BytesIO(output.getvalue()), mimetype='image/png')

@app.route('/predict_expenses', methods=['POST'])
def predict_expenses():
    # Load the expense data from the database
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    
    user_id = session.get('user_id')

    # Query to get expenses for the logged-in user
    cursor.execute("""
        SELECT amount, date AS month
        FROM income_expense
        WHERE type = 'expense' AND user_id = %s
    """, (user_id,))

    data = cursor.fetchall()
    cursor.close()
    connection.close()

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        return render_template('expensepredicitor.html', message="No expense data available for prediction.")

    # Convert 'month' to datetime
    df['month'] = pd.to_datetime(df['month'])

    # Create an ordinal column from the 'month'
    df['month_ordinal'] = df['month'].dt.to_period('M').apply(lambda x: x.to_timestamp().toordinal())

    # Prepare data for training
    X = df['month_ordinal'].values.reshape(-1, 1)
    y = df['amount'].values

    # Calculate total expenses
    total_expense = df['amount'].sum()

    # Polynomial feature transformation
    degree = 3  # Adjust the degree as needed
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X)

    # Normalize the target variable
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))

    # Use Ridge regression instead of LinearRegression to reduce overfitting
    ridge_model = Ridge(alpha=1.0)  # Adjust alpha value as necessary
    ridge_model.fit(X_train_poly, y_scaled)

    # Predict future expenses (2 months)
    last_month_ordinal = df['month_ordinal'].max()
    future_months = pd.date_range(start=df['month'].max() + pd.offsets.MonthBegin(1), periods=2, freq='M')
    future_ordinal = np.array([month.toordinal() for month in future_months]).reshape(-1, 1)
    future_ordinal_poly = poly.transform(future_ordinal)

    # Predict future expenses and convert back to original scale
    future_predictions_scaled = ridge_model.predict(future_ordinal_poly)
    future_predictions = scaler.inverse_transform(future_predictions_scaled)

    # Prepare textual summaries
    past_expenses = df[['month', 'amount']].copy()
    future_predictions_df = pd.DataFrame({
        'month': future_months,
        'predicted_amount': future_predictions.flatten()
    })

    # Convert to a readable format
    past_expenses['month'] = past_expenses['month'].dt.strftime('%B %Y')
    future_predictions_df['month'] = future_predictions_df['month'].dt.strftime('%B %Y')

    # Convert to lists of tuples for easy rendering in HTML
    past_expenses_list = list(past_expenses.itertuples(index=False, name=None))
    future_predictions_list = list(future_predictions_df.itertuples(index=False, name=None))

    # Combine actual data and future predictions for plotting
    combined_data = pd.concat([df[['month', 'month_ordinal', 'amount']], future_predictions_df.assign(amount=np.nan)], ignore_index=True)

    # Convert the 'month' column back to datetime if it's been changed to string format
    combined_data['month'] = pd.to_datetime(combined_data['month'])
    future_predictions_df['month'] = pd.to_datetime(future_predictions_df['month'])

    # Plot the results
    img = BytesIO()
    plt.figure(figsize=(12, 6))
    plt.bar(df['month'], df['amount'], label='Actual', color='blue', width=10)  # Adjust bar width for visibility

    # Filter combined_data to remove rows with NaN in 'month_ordinal'
    valid_combined_data = combined_data.dropna(subset=['month_ordinal'])

    plt.plot(valid_combined_data['month'], ridge_model.predict(poly.transform(valid_combined_data['month_ordinal'].values.reshape(-1, 1))), label=f'Polynomial Ridge Regression (Degree {degree}) Prediction', color='red', linestyle='--')
    plt.scatter(future_predictions_df['month'], future_predictions_df['predicted_amount'], color='green', label='Future Predictions', marker='x')
    plt.xlabel('Month')
    plt.ylabel('Expense Amount')
    plt.title(f'Actual vs. Predicted Monthly Expenses - Polynomial Ridge Regression (Degree {degree})')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(img, format='jpg')
    img.seek(0)

    # Ensure static directory exists
    static_folder = os.path.join(os.path.dirname(__file__), 'static', 'user_graphs')
    os.makedirs(static_folder, exist_ok=True)

    # Save image with a unique name for each user
    img_path = os.path.join(static_folder, f'expense_prediction_{user_id}.jpg')
    with open(img_path, 'wb') as f:
        f.write(img.getvalue())

    # Prepare the summary text
    summary = f"Total expenses so far: ${total_expense:.2f}. Predicted expenses for the next two months: {future_predictions[0][0]:.2f} and {future_predictions[1][0]:.2f}."

    # Render the template with textual summaries and the graph
    return render_template('expensepredicitor.html',
                           prediction_image=url_for('static', filename=f'user_graphs/expense_prediction_{user_id}.jpg'),
                           summary=summary,
                           past_expenses_list=past_expenses_list,
                           future_predictions_list=future_predictions_list,
                           message="Expense Prediction Results")

# Function to calculate future value
def future_value(invested_monthly, yrs, annual_roi=12):
    compounded_roi = annual_roi / 100 / 12
    fv = float(invested_monthly) * ((1 + compounded_roi) ** (yrs * 12) - 1) * (1 + compounded_roi) / compounded_roi
    fv = round(fv, 0)
    return fv

# Function to calculate total invested amount
def total_invested(invested_monthly, yrs):
    total_money = invested_monthly * 12 * yrs
    total_money = round(total_money, 0)
    return total_money
    
# Route to handle form submission
@app.route('/suggest', methods=['POST'])
def suggest():
    amount = request.form['income']
    age = request.form['age']
    occupation = request.form['occupation']
    str1 = ''
    
    if not amount or not age:
        str1 += "Dear User,\nPlease fill all the fields.\n\nYou may press RESET to reset fields."
    elif not isfloat(amount) or not isfloat(age):
        str1 += "Dear User,\nPlease enter appropriate values for amount and age.\n\n\t>> Income    =>    Integer or Float\n\t>> Age         =>    Preferably an Integer"
    elif float(amount) < 0 or int(age) <= 0:
        str1 += "Dear User,\nPlease enter a positive value for amount and age.\n\nYou may press RESET to reset fields."
    else:
        age = int(age)
        amount = float(amount)
        per50 = float(amount) * 0.5
        per40 = float(amount) * 0.4
        per30 = float(amount) * 0.3
        per20 = float(amount) * 0.2
        
        per50 = str(round(per50, 0))
        per40 = str(round(per40, 0))
        per30 = str(round(per30, 0))
        per20 = str(round(per20, 0))

        if age < 1 or age > 130:
            str1 += "Dear User,\nPlease enter an appropriate age.\nAge should be between 1 year and 130 years.\n\nYou may press RESET to reset fields."
        else:
            if age < 18:
                str1 += "Dear user,\nAs your age is below 18 years,\nIt won't be possible for you to invest in Stocks or Mutual Funds.\nBut you may study about stock market to get a basic idea about the same.\n\nYou may read the following books to increase your knowledge.\n\n1. The Intelligent Investor\n2. Rich Dad Poor Dad"
            else:
                if 18 <= age <= 35:
                    str1 += "Dear user,\nAs you are young, we recommend you the following investment strategies.\n\n>> 50% - For your needs (food, rent, EMI, etc.)\n\t[50 %    =>    ~ " + per50 + "    INR]\n\n>> 30% - For your wants (vacations, gadgets, etc.)\n\t[30 %    =>    ~ " + per30 + "    INR]\n\n>> 20% - Savings and Investments (Stocks, Mutual Funds, FD, etc.)\n\t[20 %    =>    ~ " + per20 + "    INR]\n"
                    str1 += "\n\n>> If you follow this Financial Discipline, \nEstimated Returns (at 12% Compound Interest) :"
                    invested = float(amount) * 0.2
                    invested = round(invested, 0)
                    str1 += "\nInvested/month      =>      " + str(invested) + " INR"
                    str1 += "\n\nPeriod\tInvested (INR)\t\tFuture Value (INR)\n---------------------------------------------------------------"
                    str1 += "\n2 yrs\t~ " + str(total_invested(invested, 2)) + "\t\t~ " + str(future_value(invested, 2))                    
                    str1 += "\n5 yrs\t~ " + str(total_invested(invested, 5)) + "\t\t~ " + str(future_value(invested, 5))
                    str1 += "\n10 yrs\t~ " + str(total_invested(invested, 10)) + "\t\t~ " + str(future_value(invested, 10))
                elif age > 35:
                    str1 += "Dear user,\nAs you are elder, we recommend you the following investment strategies.\n\n>> 40% - For your needs (food, rent, EMI, etc.)\n\t[40 %    =>    ~ " + per40 + "    INR]\n\n>> 20% - For your wants (vacations, gadgets, etc.)\n\t[20 %    =>    ~ " + per20 + "    INR]\n\n>> 40% - Savings and Investments (Stocks, Mutual Funds, FD, etc.)\n\t[40 %    =>    ~ " + per40 + "    INR]\n"
                    str1 += "\n\n>> If you follow this Financial Discipline, \nEstimated Returns (at 12% Compound Interest) :"
                    invested = float(amount) * 0.4
                    invested = round(invested, 0)
                    str1 += "\nInvested/month      =>      " + str(invested) + " INR"
                    str1 += "\n\nPeriod\tInvested (INR)\t\tFuture Value (INR)\n---------------------------------------------------------------"
                    str1 += "\n2 yrs\t~ " + str(total_invested(invested, 2)) + "\t\t~ " + str(future_value(invested, 2))                    
                    str1 += "\n5 yrs\t~ " + str(total_invested(invested, 5)) + "\t\t~ " + str(future_value(invested, 5))
                    str1 += "\n10 yrs\t~ " + str(total_invested(invested, 10)) + "\t\t~ " + str(future_value(invested, 10))

            if occupation == "Student":
                str1 += "\n\n\n>> Self Investment is the Best Investment.\nAs you are a student, you may invest your time and energy in learning via various resources such as Online Courses.\nWe recommend checking out courses on the following sites:\n1. www.coursera.org\n2. www.udemy.com"
            elif occupation == "Employee":
                str1 += "\n\n\n>> Self Investment is the Best Investment.\nAs you are an employee, you may invest your time and energy in learning via various resources, reading books.\nWe recommend checking out courses on the following sites:\n1. www.coursera.org\n2. www.udemy.com\n\n>> This surely increases your chances of promotion and gives the most returns!"
            elif occupation == "Business":
                str1 += "\n\n\n>> Self Investment is the Best Investment.\nAs you are into business, you may invest your time in reading books which help grow your business.\nWe recommend checking out the following books:\n1. Think and Grow Rich\n2. Zero to One\n3. Rich Dad Poor Dad\n\n>> This surely increases your chances of growing your business and gives the most returns!"
            elif occupation == "Housemaker":
                str1 += "\n\n\n>> Self Investment is the Best Investment.\nAs you are a housemaker, you may invest your time in learning new skills at home.\nYou may help people through these skills via social media.\nHence, you can earn money via Digital Marketing.\nWe recommend you to learn about:\n1. Digital Marketing\n2. Adsense\n3. Blogging\n\n>> This surely increases your chances of improving your skills, make money as well as help others!"
            elif occupation == "Other":
                str1 += "\n\n\n>> Self Investment is the Best Investment.\nWe recommend you to invest your time and energy to learn new skills and try to be a better person.\n\nYou may read the following books to be a better version of yourself!\n1. Getting Things Done\n2. The 7 Habits of Highly Effective People\n3. Think and Grow Rich"

    return render_template('investmentrecomaindation.html', result=str1)

# Helper function to check if a string can be converted to a float
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

@app.route('/investmentrecommaindation', methods=['POST'])
def investmentrecommaindation():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    user_id = session['user_id']
    return render_template('investmentrecomaindation.html',id=user_id)
@app.route('/investrec', methods=['POST'])
def investrec():
    # Define a default strategy to pass to the template
    default_strategy = {"strategy": "No strategy available", "allocations": {}}

    # Retrieve user_id or other necessary data as needed
    user_id = request.form.get('user_id', None)
    
    # Pass the default strategy to the render_template function
    return render_template('S.html', id=user_id, strategy=default_strategy)

# Load the model and scaler from the uploaded files
model_path = "investment_recommender_model.pkl"
scaler_path = "scaler.pkl"

# Load the pre-trained model and scaler
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

@app.route('/invest', methods=['POST'])
def invest():
    try:
        # Get input data from the form
        age = float(request.form['age'])
        income = float(request.form['income'])
        savings = float(request.form['savings'])
        risk_tolerance = int(request.form['risk_tolerance'])

        # Create DataFrame for new input (Assuming the same as before)
        new_data = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'Current Savings': [savings],
            'Risk Tolerance': [risk_tolerance]
        })

        # Scale the input data
        new_data_scaled = scaler.transform(new_data)

        # Make prediction
        prediction = model.predict(new_data_scaled)
        prediction_df = pd.DataFrame(prediction, columns=['Fixed Deposit (%)', 'Mutual Funds (%)', 'Bonds (%)', 'Stocks (%)'])

        # Extract the results
        result = prediction_df.iloc[0].to_dict()

        # Generate Pie Chart
        labels = ['Fixed Deposit', 'Mutual Funds', 'Bonds', 'Stocks']
        sizes = [result['Fixed Deposit (%)'], result['Mutual Funds (%)'], result['Bonds (%)'], result['Stocks (%)']]
        colors = ['#f1c40f', '#2ecc71', '#3498db', '#e74c3c']

        # Create the figure
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save it to a bytes buffer
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Encode it as a base64 string
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        # Pass the chart and result to the HTML template
        return render_template('investmentrecomaindation.html', result=result, plot_url=plot_url)

    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/stockrecommendation', methods=['GET', 'POST'])
def stockrecommendation():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    user_id = session['user_id']
    return render_template('stock_recommendation.html',id=user_id)
# Define your Finnhub API key
FINNHUB_API_KEY = 'crqpmfhr01qq1umns3cgcrqpmfhr01qq1umns3d0'

# Updated list of stock tickers
tickers = [
    'AAPL',  # Apple Inc.
    'TSLA',  # Tesla Inc.
    'GOOGL', # Alphabet Inc. (Google)
    'AMZN',  # Amazon.com Inc.
    'MSFT',  # Microsoft Corporation
    'NFLX',  # Netflix Inc.
    'NVDA',  # NVIDIA Corporation
    'FB',    # Meta Platforms Inc.
    'INTC',  # Intel Corporation
    'BABA',  # Alibaba Group Holding Limited
    'JPM',   # JPMorgan Chase & Co.
    'V',     # Visa Inc.
    'MA',    # Mastercard Inc.
    'DIS',   # Walt Disney Co.
    'PYPL',  # PayPal Holdings Inc.
    'CSCO',  # Cisco Systems Inc.
    'ADBE',  # Adobe Inc.
    'ORCL',  # Oracle Corporation
    'PEP',   # PepsiCo Inc.
    'NKE',   # Nike Inc.
]

def fetch_finnhub_data(ticker):
    url = f'https://finnhub.io/api/v1/quote'
    params = {
        'symbol': ticker,
        'token': FINNHUB_API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    try:
        current_price = data['c']
        open_price = data['o']
        high_price = data['h']
        low_price = data['l']
    except KeyError:
        current_price, open_price, high_price, low_price = None, None, None, None
    
    return {
        'current_price': current_price,
        'open_price': open_price,
        'high_price': high_price,
        'low_price': low_price
    }

def apply_fuzzy_logic(data):
    fuzzy_results = []
    
    for ticker, stock_data in data.items():
        if stock_data['current_price'] is None:
            continue
        
        recommendation = stock_data['current_price']
        
        # Normalization
        old_value = recommendation
        old_min = 50   # Assume $50 as minimum stock price
        old_max = 2000 # Assume $2000 as maximum stock price
        new_min = 0
        new_max = 1
        normalized_value = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        
        fuzzy_normalized_value = 1 - normalized_value
        fuzzy_normalized_value = round(fuzzy_normalized_value, 2)
        
        # Fuzzy categories
        if 0 <= fuzzy_normalized_value <= 0.2:
            fuzzy = "Sell"
        elif 0.2 < fuzzy_normalized_value <= 0.4:
            fuzzy = "Underperforming"
        elif 0.4 < fuzzy_normalized_value <= 0.6:
            fuzzy = "Hold"
        elif 0.6 < fuzzy_normalized_value <= 0.8:
            fuzzy = "Buy"
        elif fuzzy_normalized_value > 0.8:
            fuzzy = "Strong Buy"
        
        fuzzy_results.append([ticker, stock_data['current_price'], fuzzy_normalized_value, fuzzy])
    
    fuzzy_df = pd.DataFrame(fuzzy_results, columns=['Company', 'Current Price', 'Normalized Value', 'Fuzzy Decision'])
    return fuzzy_df
def personalize_recommendations(fuzzy_df, income, risk_tolerance, age, savings):
    personalized_recommendations = []

    # Ensure the user is of valid age
    if age < 18:
        raise ValueError("Age must be greater than or equal to 18.")

    for index, row in fuzzy_df.iterrows():
        fuzzy_decision = row['Fuzzy Decision']
        current_price = row['Current Price']

        # Basic filtering logic based on risk tolerance
        if risk_tolerance == 'low':
            if fuzzy_decision == 'Strong Buy':
                continue  # Skip Strong Buy for low-risk
            elif fuzzy_decision in ['Buy', 'Hold']:
                # Check if the stock price is affordable based on income
                if current_price > income * 0.1:  # Example: can't afford to invest 10% of income
                    continue

        elif risk_tolerance == 'high':
            if fuzzy_decision in ['Underperforming', 'Sell']:
                continue  # Skip safe options for high-risk tolerant users

        elif risk_tolerance == 'medium':
            # Medium risk tolerance can consider all recommendations but may include age as a factor
            if age > 50 and fuzzy_decision in ['Strong Buy', 'Buy']:
                # Suggest stocks that have stable performance for older users
                if current_price > 100:  # Example threshold
                    continue  # Skip high-priced stocks for older users

        # Add row to personalized recommendations if it passes the filters
        personalized_recommendations.append(row)

    # Create a DataFrame for personalized recommendations
    personalized_df = pd.DataFrame(personalized_recommendations, columns=fuzzy_df.columns)
    return personalized_df


@app.route('/stockrecommender', methods=['GET', 'POST'])
def stockrecommender():
    recommendations = []
    if request.method == 'POST':
        income = float(request.form['income'])
        risk_tolerance = request.form['risk_tolerance']
        age = int(request.form['age'])
        savings = float(request.form['savings'])

        stock_data = {}
        for ticker in tickers:
            stock_data[ticker] = fetch_finnhub_data(ticker)

        fuzzy_df = apply_fuzzy_logic(stock_data)
        recommendations = personalize_recommendations(fuzzy_df, income, risk_tolerance, age, savings)

        # Convert DataFrame to a list of dictionaries
        recommendations = recommendations.to_dict(orient='records')

    return render_template('stock_recommendation.html', recommendations=recommendations)

# Load the trained model, column transformer, and scaler
classifier = joblib.load('random_forest_model.pkl')
column_transformer = joblib.load('column_transformer.pkl')
scalersh = joblib.load('scalershu.pkl')

# Define investment strategies for each cluster
investment_strategies = {
    0: {"strategy": "Low-risk", "allocations": {"Fixed Deposit": 50.0, "Bonds": 40.0, "Stocks": 10.0, "Mutual Funds": 0.0, "Real Estate": 0.0}},
    1: {"strategy": "Balanced", "allocations": {"Stocks": 30.0, "Bonds": 30.0, "Mutual Funds": 30.0, "Real Estate": 10.0, "Fixed Deposit": 0.0}},
    2: {"strategy": "High-risk", "allocations": {"Stocks": 50.0, "Bonds": 20.0, "Mutual Funds": 30.0, "Real Estate": 0.0, "Fixed Deposit": 0.0}},
    3: {"strategy": "Growth", "allocations": {"Stocks": 40.0, "Real Estate": 40.0, "Mutual Funds": 20.0, "Bonds": 0.0, "Fixed Deposit": 0.0}}
}

def recommend_investment_strategy(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    transformed_input = column_transformer.transform(input_array)
    scaled_input = scalersh.transform(transformed_input)
    predicted_cluster = classifier.predict(scaled_input)[0]
    strategy = investment_strategies.get(predicted_cluster, {"strategy": "No strategy available", "allocations": {}})
    return predicted_cluster, strategy

def create_pie_chart(allocations):
    labels = list(allocations.keys())
    sizes = list(allocations.values())
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    
    # Save pie chart to a bytes buffer and convert to base64 for HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return image_base64

@app.route('/shuban', methods=['POST'])
def shuban():
    try:
        input_data = [
            int(request.form['age']),
            float(request.form['income']),
            float(request.form['fixed_expenses']),
            float(request.form['variable_expenses']),
            float(request.form['savings_rate']),
            float(request.form['disposable_income']),
            float(request.form['total_savings']),
            float(request.form['stocks']),
            float(request.form['mutual_funds']),
            float(request.form['bonds']),
            float(request.form['real_estate']),
            float(request.form['fixed_deposits']),
            float(request.form['other_investments']),
            float(request.form['debt']),
            float(request.form['monthly_debt_payment']),
            float(request.form['insurance']),
            float(request.form['short_term_goals']),
            float(request.form['medium_term_goals']),
            float(request.form['long_term_goals']),
            request.form['risk_tolerance'],
            int(request.form['investment_horizon']),
            int(request.form['transaction_history'])
        ]
    except KeyError as e:
        return Response(f"Missing form field: {str(e)}", status=400)

    # Predict cluster and get recommendation
    predicted_cluster, strategy = recommend_investment_strategy(input_data)
    
    # Generate pie chart for the allocation strategy
    pie_chart = create_pie_chart(strategy['allocations'])

    # Render the result in a template with strategy and chart
    return render_template('S.html', 
                           predicted_cluster=predicted_cluster, 
                           strategy=strategy, 
                           pie_chart=pie_chart)
if __name__ == '__main__':
    app.run(debug=True)
