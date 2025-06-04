#!/bin/bash
# Test script for AIMS

source venv/bin/activate
pytest tests/ -v --tb=short
