from django.shortcuts import render, HttpResponse, redirect

from accounts.models import UserAccount
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings

from accounts.models import UserAccount
from .forms import UserAccountForm

from .utils import generate_ref_code, send_email
from .emails import send_verification_email
from .emails import send_verification_email, send_change_password_email

def register_page(request):
    domain_name = request.build_absolute_uri('/')[:-1]

    if request.method == "POST":
        form = UserAccountForm(request.POST)

        if form.is_valid():
            user = form.save()

            # Send verification email.
            send_verification_email(email=user.email, auth_token=user.email_auth_token, domain_name=domain_name)
            # send_verification_sms ...

            messages.success(request, "Registration successful!")
            return render(request, "accounts/check_your_email.html", {'email': user.email})
            # return redirect("/")
        
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    # print(request, f"{field.capitalize()}: {error}")
                    messages.error(request, f"{error}")
                    
            return render(request, "accounts/register.html", { 'form': form })
                
    user_form = UserAccountForm()

    context = {
        'form': user_form,
    }

    return render(request, "accounts/register.html", context)


def login_page(request):
    # user = request.user
    domain_name = request.build_absolute_uri('/')[:-1]
    if request.method == "POST":
        email = request.POST['email']
        password = request.POST['password']

        user = authenticate(request, username=email, password=password)

        if user is not None:
            if not user.verified_email:
                # Send verification codes again
                send_verification_email(email=user.email, auth_token=user.email_auth_token, domain_name=domain_name)
                messages.info(request, f'This Accounts email is not verified')
                return render(request, "accounts/check_your_email.html", {'email': user.email})

            login(request, user)
            return redirect("/")

        else:
            messages.error(request, f'The email or password is incorrect')

    return render(request, "accounts/login.html")


def logout_page(reqeust):
    logout(reqeust)
    return redirect("/accounts/login/")


def verify_email_page(request, auth_token):
    try:
        user = UserAccount.objects.get(email_auth_token=auth_token)
    except:
        return redirect("/accounts/error/")


    user.verified_email = True
    user.save()

    messages.success(request, f'Your email has been verified!')
    return redirect("/accounts/login/")


def forgot_password(request):
    domain_name = request.build_absolute_uri('/')[:-1]

    if request.method == "POST":
        email = request.POST["email"]
        try:
            user = UserAccount.objects.get(email=email)
        except:
            messages.info(request, f'Incorrect email.')
            return render(request, "accounts/login.html")
        
        send_change_password_email(email=user.email, auth_token=user.email_auth_token, domain_name=domain_name)
        print("success")
        return render(request, "accounts/check_your_email.html", {'email': user.email})

    return render(request, "accounts/forgot_password.html")

# @login_required
def change_password(request, email, auth_token):
    try:
        user = UserAccount.objects.get(email=email)
        if not user.verified_email:
            messages.info(
                request, f'This email is not verified, register again to get verification emial.')
            return redirect("/accounts/register/")
    except:
        messages.info(request, f'Incorrect email.')
        return render(request, "accounts/login.html")

    if user.email_auth_token != auth_token:
        return redirect("accounts/error/")

    if request.method == "POST":
        new_password = request.POST["new_password"]
        conform_new_password = request.POST["confirm_password"]

        # Gate keeping.
        if new_password != conform_new_password:
            messages.error(
                request, "Passwords didn't match, Please try again.")
            return redirect(f"/accounts/change_password/{user.email}/{user.email_auth_token}/")
        if len(new_password) < 8:
            messages.error(
                request, "Passwords must contain at least 8 characters.")
            return redirect(f"/accounts/change_password/{user.email}/{user.email_auth_token}/")

        # All Correct.

        # OK, Change password.
        user.set_password(new_password)
        user.auth_token = generate_ref_code()
        user.save()

        messages.success(request, "Password changed successfully!")
        return redirect('/')

    return render(request, "accounts/change_password.html")


def error(request):
    return render(request, "accounts/error.html")

