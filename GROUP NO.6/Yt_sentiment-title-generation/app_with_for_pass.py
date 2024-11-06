from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mail import Mail, Message
from flask_mysqldb import *
import secrets
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your secret key

# Configuration for Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'sskadam2735@gmail.com'  # Replace with your Gmail address
app.config['MAIL_PASSWORD'] = 'xdnqvckbeatxnfsl'     # Replace with your 16-character App Password
mail = Mail(app)

# Database connection
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'user_database'
mysql = MySQL(app)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Forgot Password Page
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        token = secrets.token_urlsafe(16)  # Generate a secure token
        expiration = datetime.utcnow() + timedelta(hours=1)  # Token valid for 1 hour

        # Store the token in the database
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO password_resets (email, token, expiration) VALUES (%s, %s, %s)",
                       (email, token, expiration))
        mysql.connection.commit()
        cursor.close()

        # Send the reset email with a friendly message and an image
        reset_link = url_for('reset_password', token=token, _external=True)
        msg = Message('Password Reset Request', sender='your_email@gmail.com', recipients=[email])

        msg.body = f"""
        Hi,

        It looks like you forgot your password, but don't worry! Just click the link below to reset it:

        {reset_link}

        If you didn't request this, feel free to ignore this email.

        Best regards,
        APSIT-TE-DS
        """

        msg.html = f"""
        <p>Hi,</p>
        <p>It looks like you forgot your password, but don't worry! Just click the link below to reset it:</p>
        <a href="{reset_link}">Reset Password</a>
        <p>If you didn't request this, feel free to ignore this email.</p>
        <p>Best regards,</p>
        <p>APSIT-TE-DS</p>
        <img src="cid:image1" alt="Reset Password" style="width: 650px; height: 500px;">
        """

        ## Attach the image from the static directory
        with app.open_resource("static/image.png") as img:
            msg.attach("image.png", "image/png", img.read(), headers={'Content-ID': '<image1>'})

        mail.send(msg)
        flash('A password reset link has been sent to your email.', 'success')
        print("Resent link Sent, Check Mail")#For Debugging
        return redirect(url_for('index'))

    return render_template('forgot_password.html')


# Reset Password Page
@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if request.method == 'POST':
        new_password = request.form['new_password']

        # Validate the token
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT email FROM password_resets WHERE token = %s AND expiration > NOW()", (token,))
        result = cursor.fetchone()

        if result:
            email = result[0]

            # Update the user's password in the database
            cursor.execute("UPDATE users SET password = %s WHERE email = %s", (new_password, email))
            mysql.connection.commit()

            # Delete the token from the database
            cursor.execute("DELETE FROM password_resets WHERE token = %s", (token,))
            mysql.connection.commit()

            flash('Your password has been reset successfully.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid or expired token.', 'danger')
            return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=8000)
