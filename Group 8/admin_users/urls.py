from django.urls import path

from . import views
urlpatterns = [
    path('', views.home, name="home"),
    path('check_application/<str:application_id>', views.check_application, name="admin_check_application"),
    path('courses/', views.courses, name="courses"),
    path('edit_course/<str:course_id>', views.edit_courses, name="edit_course"),
    path('create_course/', views.create_courses, name="create_course"),

    path('unverified_accounts/', views.unverified_accounts, name="unverified_accounts"),
    path('unverified_accounts/<str:account_id>/', views.unverified_accounts, name="unverified_accounts"),

    path('verified_accounts/', views.verified_accounts, name="verified_accounts"),
    path('verified_accounts/<str:account_id>/', views.verified_accounts, name="verified_accounts"),
]
