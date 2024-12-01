from flask import Flask, render_template, request, redirect, url_for, session
import mysql.connector
from mysql.connector import errorcode
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secure_random_secret_key_here'  # Replace with a secure secret key

# Database configuration
db_config = {
    'user': 'root',                 # Replace with your MySQL username
    'password': 'Parthavi@1204',    # Replace with your MySQL password
    'host': 'localhost',
    'database': 'digifnance'
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)

@app.route('/')
def index():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))  # Redirect to dashboard if logged in
    return render_template('login.html')  # Ensure login.html exists in the templates folder

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user and check_password_hash(user['password'], password):
        session['logged_in'] = True
        session['user_id'] = user['id']
        return redirect(url_for('dashboard'))  # Redirect to the dashboard route
    else:
        return render_template('login.html', error='Invalid credentials')  # Ensure login.html exists in the templates folder

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    # Use 'pbkdf2:sha256' for better security
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('INSERT INTO users (username, email, password) VALUES (%s, %s, %s)',
                       (username, email, hashed_password))
        conn.commit()
    except mysql.connector.Error as err:
        print(err)
        return render_template('login.html', error='Registration failed. Email may already be in use.')
    finally:
        cursor.close()
        conn.close()

    return redirect(url_for('index'))  # Redirect to the index route

@app.route('/dashboard')
def dashboard():
    if session.get('logged_in'):
        return render_template('hello.html')  # Ensure index.html exists in the templates folder
    else:
        return redirect(url_for('index'))  # Redirect to login if not logged in

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_id', None)
    return redirect(url_for('index'))  # Redirect to the index route

if __name__ == "__main__":  # Correct condition for running the Flask app
    app.run(debug=True)
