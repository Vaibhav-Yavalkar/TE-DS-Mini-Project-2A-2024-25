from flask import Flask, request, redirect, render_template, url_for, send_file
import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Your existing database configuration
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
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)  # Use dictionary cursor

    # Fetch total balance
    cursor.execute("SELECT SUM(amount) AS total_balance FROM income_expense WHERE type='income'")
    income_result = cursor.fetchone()
    total_income = income_result['total_balance'] or 0

    cursor.execute("SELECT SUM(amount) AS total_expense FROM income_expense WHERE type='expense'")
    expense_result = cursor.fetchone()
    total_expense = expense_result['total_expense'] or 0

    balance = total_income - total_expense

    # Fetch expenses by category
    cursor.execute("""
        SELECT category, SUM(amount) AS total_amount
        FROM income_expense
        WHERE type='expense'
        GROUP BY category
    """)
    expenses_by_category = cursor.fetchall()

    # Fetch recent transactions
    cursor.execute("""
        SELECT purpose, category, amount, date
        FROM income_expense
        ORDER BY date DESC
        LIMIT 5
    """)
    recent_transactions = cursor.fetchall()

    cursor.close()
    connection.close()

    return render_template('index.html', balance=balance, expenses_by_category=expenses_by_category, recent_transactions=recent_transactions)

@app.route('/delete_transaction/<int:transaction_id>', methods=['POST'])
def delete_transaction(transaction_id):
    connection = create_connection()
    cursor = connection.cursor()

    try:
        query = "DELETE FROM income_expense WHERE id = %s"
        cursor.execute(query, (transaction_id,))
        connection.commit()
        print(f"Transaction with ID {transaction_id} deleted.")
    except Error as e:
        print(f"The error '{e}' occurred")
    finally:
        cursor.close()
        connection.close()

    return redirect(url_for('incexp'))

@app.route('/incexp')
def incexp():
    """Render the main page and show transactions."""
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM income_expense")
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
    """Serve the chart image."""
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM income_expense")
    transactions = cursor.fetchall()
    cursor.close()
    connection.close()

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
    
    return send_file(img, mimetype='image/png')

@app.route('/add_income_expense', methods=['POST'])
def add_income_expense():
    amount = request.form.get('amount')
    category = request.form.get('category')
    date = request.form.get('date')
    purpose = request.form.get('purpose')  # Make sure you include purpose in the form
    record_type = request.form.get('type')

    connection = create_connection()
    cursor = connection.cursor()

    try:
        query = """
        INSERT INTO income_expense (amount, category, date, purpose, type)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (amount, category, date, purpose, record_type))
        connection.commit()
        return redirect('/')
    except Error as e:
        print(f"The error '{e}' occurred")
        return "An error occurred while adding the record."
    finally:
        cursor.close()
        connection.close()

if __name__ == '__main__':
    app.run(debug=True)
