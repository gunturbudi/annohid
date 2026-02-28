from django.apps import AppConfig


def set_sqlite_wal_mode(sender, connection, **kwargs):
    """Enable WAL mode for SQLite to prevent 'database is locked' errors."""
    if connection.vendor == 'sqlite':
        cursor = connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL;')
        cursor.execute('PRAGMA synchronous=NORMAL;')
        cursor.execute('PRAGMA busy_timeout=30000;')
        cursor.execute('PRAGMA temp_store=MEMORY;')


class AnnotationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'annotation'

    def ready(self):
        from django.db.backends.signals import connection_created
        connection_created.connect(set_sqlite_wal_mode)
