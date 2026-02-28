#!/bin/bash

# Exit on error
set -e

echo "Starting Django Medical Annotation System..."

# Wait for database to be ready
if [ "$DB_ENGINE" = "mysql" ]; then
    echo "Waiting for MySQL to be ready..."
    MAX_RETRIES=30
    RETRY_COUNT=0
    until python -c "
import MySQLdb
MySQLdb.connect(
    host='${DB_HOST:-db}',
    user='${DB_USER:-annotation_user}',
    passwd='${DB_PASSWORD:-annotation_pass}',
    db='${DB_NAME:-annotation_db}',
    port=int('${DB_PORT:-3306}')
)
print('MySQL is ready!')
" 2>/dev/null; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "ERROR: MySQL not ready after $MAX_RETRIES attempts. Exiting."
            exit 1
        fi
        echo "MySQL not ready yet (attempt $RETRY_COUNT/$MAX_RETRIES)... waiting 2s"
        sleep 2
    done
else
    echo "Using SQLite - checking database access..."
    sleep 2
fi

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate --noinput

# Create superuser if it doesn't exist
echo "Checking for superuser..."
python manage.py shell << EOF
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    print("Creating admin user...")
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123', role='admin')
    print("Admin user created successfully!")
else:
    print("Admin user already exists.")
EOF

# Create annotator users if they don't exist
echo "Checking for annotator users..."
python manage.py shell << EOF
from django.contrib.auth import get_user_model
User = get_user_model()

annotators = [
    ('annotator1', 'ann123'),
    ('annotator2', 'ann234'),
    ('annotator3', 'ann345'),
    ('annotator4', 'ann456'),
    ('annotator5', 'ann707'),
]

for username, password in annotators:
    if not User.objects.filter(username=username).exists():
        User.objects.create_user(
            username=username,
            email=f'{username}@example.com',
            password=password,
            role='annotator'
        )
        print(f"Created user: {username}")
    else:
        print(f"User {username} already exists.")
EOF

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear || true

echo "Initialization complete!"
echo "="*50
echo "Django Medical Annotation System is ready!"
echo "Access the application at: http://localhost:8810"
echo "Admin credentials: admin / admin123"
echo "Annotator credentials: annotator1-5 / ann123-ann707"
echo "="*50

# Execute the main command
exec "$@"
