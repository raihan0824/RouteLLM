#!/bin/sh

# Start the first application in the background
python -m routellm.openai_server --routers mf --strong-model openai/qwen/qwen2-vl-72b-instruct --weak-model openai/gotocompany/gemma2-9b-cpt-sahabatai-v1-instruct