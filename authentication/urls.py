from django.urls import path
from . import views
from . import gamification_views

app_name = 'authentication'

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('profile/', views.profile_view, name='profile'),
    path('update-activity/', views.update_activity, name='update_activity'),

    # User management URLs
    path('users/', views.user_management, name='user_management'),
    path('users/create/', views.create_user, name='create_user'),
    path('users/quick-create-annotator/', views.quick_create_annotator, name='quick_create_annotator'),
    path('reset-all-data/', views.reset_all_data, name='reset_all_data'),
    path('users/<int:user_id>/edit/', views.edit_user, name='edit_user'),
    path('users/<int:user_id>/password/', views.change_user_password, name='change_user_password'),
    path('users/<int:user_id>/delete/', views.delete_user, name='delete_user'),
    
    # Gamification URLs
    path('leaderboard/', gamification_views.leaderboard_view, name='leaderboard'),
    path('achievements/', gamification_views.achievements_view, name='achievements'),
    path('points-history/', gamification_views.points_history_view, name='points_history'),
    path('gamification-config/', gamification_views.gamification_config_view, name='gamification_config'),
    path('api/gamification/award-points/', gamification_views.award_points_api, name='api_award_points'),
]