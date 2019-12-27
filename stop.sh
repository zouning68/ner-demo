#!/bin/bash

SCRIPT_NAME="main.py"

ps aux | grep "${SCRIPT_NAME}" | grep -v "grep" | awk '{print $2}' | xargs kill -9

