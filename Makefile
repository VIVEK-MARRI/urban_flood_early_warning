# Makefile for Urban Flood Early Warning System

.PHONY: setup up down test lint clean help

help:
	@echo "Available commands:"
	@echo "  make setup   - Install python dependencies"
	@echo "  make up      - Start all services (Docker)"
	@echo "  make down    - Stop all services"
	@echo "  make test    - Run unit tests"
	@echo "  make lint    - Run code formatting checks"
	@echo "  make clean   - Remove temporary files"

setup:
	pip install -r api/requirements.txt
	pip install -r streamlit_app/requirements.txt
	pip install dvc pytest black pylint

up:
	docker-compose up --build -d

down:
	docker-compose down

test:
	pytest tests/ -v

lint:
	black src/ api/ airflow/ --check

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
