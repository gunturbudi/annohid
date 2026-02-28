from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.contrib.auth.forms import AdminPasswordChangeForm
from django.urls import path, reverse
from django.utils.html import format_html
from django.http import HttpResponseRedirect
from .models import User, UserSession


class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = User
        fields = ("username", "email", "role", "first_name", "last_name")


class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = User
        fields = ("username", "email", "first_name", "last_name", "is_active", "role")


class UserAdmin(BaseUserAdmin):
    add_form = CustomUserCreationForm
    form = CustomUserChangeForm
    change_password_form = AdminPasswordChangeForm
    model = User

    list_display = ("username", "email", "role", "is_active", "last_activity", "created_at", "user_actions")
    list_filter = ("role", "is_active", "created_at", "last_activity")
    search_fields = ("username", "email", "first_name", "last_name")
    ordering = ("username",)
    list_per_page = 25
    actions = ['activate_users', 'deactivate_users', 'reset_passwords']

    fieldsets = (
        (None, {"fields": ("username", "password")}),
        ("Personal info", {"fields": ("first_name", "last_name", "email")}),
        ("Permissions", {"fields": ("role", "is_active", "is_staff", "is_superuser", "groups", "user_permissions")}),
        ("Important dates", {"fields": ("last_login", "date_joined", "last_activity")}),
        ("Activity", {"fields": ("created_at", "updated_at")}),
    )

    add_fieldsets = (
        (None, {
            "classes": ("wide",),
            "fields": ("username", "email", "first_name", "last_name", "role", "password1", "password2", "is_active"),
        }),
    )

    readonly_fields = ("last_activity", "created_at", "updated_at", "last_login", "date_joined")

    def user_actions(self, obj):
        """Add action buttons for each user"""
        change_password_url = reverse('admin:auth_user_password_change', args=[obj.pk])
        return format_html(
            '<a class="button" href="{}">Change Password</a>',
            change_password_url
        )
    user_actions.short_description = "Actions"
    user_actions.allow_tags = True

    def activate_users(self, request, queryset):
        """Bulk activate users"""
        updated = queryset.update(is_active=True)
        self.message_user(request, f'{updated} users were successfully activated.')
    activate_users.short_description = "Activate selected users"

    def deactivate_users(self, request, queryset):
        """Bulk deactivate users"""
        updated = queryset.update(is_active=False)
        self.message_user(request, f'{updated} users were successfully deactivated.')
    deactivate_users.short_description = "Deactivate selected users"

    def reset_passwords(self, request, queryset):
        """Bulk reset passwords (admin will need to manually set them)"""
        count = queryset.count()
        self.message_user(request, f'Please manually reset passwords for {count} selected users using the "Change Password" link.')
    reset_passwords.short_description = "Reset passwords for selected users"


class UserSessionAdmin(admin.ModelAdmin):
    list_display = ("user", "session_key_short", "created_at", "last_activity", "ip_address", "is_active")
    list_filter = ("is_active", "created_at")
    search_fields = ("user__username", "ip_address")
    readonly_fields = ("session_key", "created_at", "last_activity")

    def session_key_short(self, obj):
        return f"{obj.session_key[:8]}..."
    session_key_short.short_description = "Session Key"


admin.site.register(User, UserAdmin)
admin.site.register(UserSession, UserSessionAdmin)
