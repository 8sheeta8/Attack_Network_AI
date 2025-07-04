#!/bin/bash
cd "$(dirname "$0")"/..
docker compose down -v
docker compose up -d
sleep 10  # 컨테이너 기동 대기
