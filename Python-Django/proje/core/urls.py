
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from sample import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index,name='index'),
    path('predictimage',views.predictimage, name='predictimage')
]

urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)

urlpatterns += static(settings.STATIC_URL, document_root = settings.STATIC_ROOT)





