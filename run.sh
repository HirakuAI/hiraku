#!/bin/bash

npm i --legacy-peer-deps
npm run build
./backend/start.sh
