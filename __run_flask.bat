@REM activate the virtual environment and run the Flask app

venv\scripts\Activate.bat && flask --app src\saferplaces_agent\agent_interface\flask_server\app.py run --debug