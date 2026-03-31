
# Azure App Service startup script
#!/bin/bash
gunicorn --bind=0.0.0.0 --timeout 600 flask_app:app