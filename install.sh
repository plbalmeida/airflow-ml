virtualenv -p python3 venv;
. venv/bin/activate;
pip install -r requirements.txt;
pre-commit install;