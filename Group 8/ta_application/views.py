from django.shortcuts import render, redirect

from django.contrib.auth.decorators import login_required

from accounts.models import UserAccount

# @login_required
def redirect_user(request):
    user = request.user
    if request.user.is_anonymous:
        return render(request, "landing.html")


    if request.user.is_anonymous:
        # User is not logged in (anonymous)
        print("User is anonymous")

    if user.user_type == UserAccount.APPLICANT:
        return redirect('/core/')
    
    elif user.user_type == UserAccount.STAFF:
        return redirect('/d_staffs/')
    
    elif user.user_type == UserAccount.COMMITTEE_MEMBER:
        return redirect('/committee_members/')

    elif user.user_type == UserAccount.INSTRUCTOR:
        return redirect('/instructors/')
    
    elif user.user_type == UserAccount.ADMIN:
        return redirect('/admin_users/')

# Sorry message for not verified users
def not_verified_user(request):
    return render(request, "not_verified_user.html")