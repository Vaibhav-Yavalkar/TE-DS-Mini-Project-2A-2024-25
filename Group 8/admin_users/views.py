from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test

from core.models import Application, Comment, Course
from .forms import CourseForm
from accounts.models import UserAccount

def is_admin(user):
    return user.user_type == user.ADMIN

@login_required
@user_passes_test(is_admin, login_url="/")
def home(request):
    application_list = Application.objects.all()

    context = {
        'application_list': application_list,
    }
    return render(request, "admin_users/home.html", context)

@login_required
@user_passes_test(is_admin, login_url="/")
def check_application(request, application_id):
    application = Application.objects.get(id=application_id)
    checklist = {
        "Bachelor's degree in a relevant field": application.box1,
        "Prior teaching experience": application.box2,
        "Ability to work collaboratively": application.box3,
        "Organizational skills": application.box4,
        "Certifications and Training": application.box5,
        "Additional Qualifications": application.box6,
        "Personal Statement/Statement of Purpose": application.box7,
        "Proficiency in specific subject areas": application.box8,
    }

    no_of_checks = 0
    if application.box1: no_of_checks += 1
    if application.box2: no_of_checks += 1
    if application.box3: no_of_checks += 1
    if application.box4: no_of_checks += 1
    if application.box5: no_of_checks += 1
    if application.box6: no_of_checks += 1
    if application.box7: no_of_checks += 1
    if application.box8: no_of_checks += 1

    if request.method == "POST":
        if "approve" in request.POST:
            application.approve_by(request.user)
            messages.success(request, "Your response was recorded!")

        if "disapprove" in request.POST:
            application.disapprove_by(request.user)
            messages.success(request, "Your response was recorded!")

        if "add_comment" in request.POST:
            message = request.POST.get("message")
            if message:
                new_comment = Comment(message=message, user=request.user, application=application)
                new_comment.save()
                messages.success(request, "Your response was recorded!")

    comment_list = Comment.objects.filter(application=application)
    context = {
        'application': application,
        'comment_list': comment_list,
        'checklist': checklist,
        'completion_percentage': (no_of_checks / 8) * 100,
    }
    return render(request, "admin_users/check_application.html", context)

@login_required
@user_passes_test(is_admin, login_url="/")
def courses(request):
    course_list = Course.objects.all()

    context = {
        'course_list': course_list,
    }
    return render(request, "admin_users/courses.html", context)


@login_required
@user_passes_test(is_admin, login_url="/")
def edit_courses(request, course_id):
    course = Course.objects.get(id=course_id)

    if request.method == "POST":
        form = CourseForm(request.POST, instance=course)
        if form.is_valid():
            form.save()
            return redirect("/admin_users/courses/")

    form = CourseForm(instance=course)
    return render(request, "admin_users/create_course.html", { 'form': form } )


@login_required
@user_passes_test(is_admin, login_url="/")
def create_courses(request):
    if request.method == "POST":
        form = CourseForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("/admin_users/")

    form = CourseForm()
    return render(request, "admin_users/create_course.html", { 'form': form } )


@login_required
@user_passes_test(is_admin, login_url="/")
def unverified_accounts(request, account_id=None):
    if account_id:
        account = UserAccount.objects.get(id=account_id)
        account.is_verified_user = True
        account.save()

    account_list = UserAccount.objects.filter(
        is_verified_user=False
        ).exclude(user_type=UserAccount.APPLICANT
        ).exclude(user_type=UserAccount.ADMIN
    )

    return render(request, "admin_users/unverified_accounts.html", { 'account_list': account_list })


@login_required
@user_passes_test(is_admin, login_url="/")
def verified_accounts(request, account_id=None):
    if account_id:
        account = UserAccount.objects.get(id=account_id)
        account.is_verified_user = False
        account.save()

    account_list = UserAccount.objects.filter(
        is_verified_user=True).exclude(user_type=UserAccount.APPLICANT)

    return render(request, "admin_users/verified_accounts.html", { 'account_list': account_list })