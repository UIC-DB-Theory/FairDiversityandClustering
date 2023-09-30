#!/bin/bash

cd adult && wget https://archive.ics.uci.edu/static/public/2/adult.zip && unzip \*.zip && python3 adult_generate.py && cd ..

cd diabetes && wget https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip && unzip \*.zip && python3 diabetes_generate.py && cd ..

cd census && python3 census_generate.py && cd ..