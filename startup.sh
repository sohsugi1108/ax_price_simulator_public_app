#!/bin/bash

# Azure Web Appの環境変数を.envファイルに書き込む
echo "APP_TITLE=$APP_TITLE" >> .env
echo "DATA_DIR=$DATA_DIR" >> .env
echo "MIN_YEAR=$MIN_YEAR" >> .env
echo "MAX_YEAR=$MAX_YEAR" >> .env
echo "PORT=$PORT" >> .env

# Streamlitアプリケーションを起動
streamlit run app_refactored.py --server.port $PORT --server.address 0.0.0.0
