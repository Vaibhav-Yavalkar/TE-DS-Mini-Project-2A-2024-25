"""
URL configuration for ta_application project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from . import views
urlpatterns = [
    path('', views.redirect_user, name="redirect_user"),

    # Applicant staff APP
    path('core/', include("core.urls")),
    # Department staff APP
    path('d_staffs/', include("d_staffs.urls")),
    # Committee members APP
    path('committee_members/', include("committee_members.urls")),
    # Instructors APP
    path('instructors/', include("instructors.urls")),
    # Admin users APP
    path('admin_users/', include("admin_users.urls")),
    
    # Sorry, Not verified
    path('not_verified_user/', views.not_verified_user ,name="not_verified_user"),

    path('accounts/', include("accounts.urls")),

    
    path('admin/', admin.site.urls),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
