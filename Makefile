#!/usr/bin/make
SHELL := /bin/bash
config = "./configs/local.yaml"


check:
	. $$(yq .local.venv configs/local.yaml)/bin/activate && \
	flake8 --exclude=cache,data,configs,examples --select=F --ignore=E,W,C

run:
	. $$(yq .local.venv configs/local.yaml)/bin/activate && ./main.py
