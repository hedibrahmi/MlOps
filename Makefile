# Déclaration des variables
PYTHON=python3
ENV_NAME=mlops_env
REQUIREMENTS=requirements.txt



install:
	pip install -r requirements.txt

prepare:
	python main.py --mode prepare

train:
	python main.py --mode train

predict:
	python main.py --mode predict

evaluate:
	python main.py --mode evaluate

save:
	python main.py --mode save

lint:
	flake8 main.py --format=html --htmldir=flake8-report

webhook:
	python webhook_server.py

all: install prepare train evaluate save lint
