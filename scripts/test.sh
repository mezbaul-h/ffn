#!/usr/bin/env bash

TESTING=true coverage run --source='.' -m pytest
coverage report
coverage xml
