from django.urls import path
from . import views
from .views import (
    register_user, 
    jwt_login_view, 
    logout_view, 
    cell_detection_view, 
    user_notes_view,
    research_analyzer_view,
    heart_disease_prediction_view,
    lung_cancer_prediction_view,
    lung_cancer_detection_view,
    auto_publish_view
)


urlpatterns = [
    # authentication endpoints
    path('register/', register_user),
    path('login/', jwt_login_view),
    path('logout/', logout_view),

    # other service endpoints
    path('mol_binding/', views.mol_binding_view),
    path('ihc_insight/', views.ihc_insight_view),
    path('cell_detection/', cell_detection_view),
    path('research_analyzer/', research_analyzer_view),
    path('heart_disease_prediction/', heart_disease_prediction_view),
    path('lung_cancer_prediction/', lung_cancer_prediction_view),
    path('lung_cancer_detection/', lung_cancer_detection_view),
    path('auto_publish/', auto_publish_view),

    # user profile endpoints
    path('user_notes/', user_notes_view),
]
