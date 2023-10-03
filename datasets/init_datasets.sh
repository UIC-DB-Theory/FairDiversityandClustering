#!/bin/bash

cd adult && wget https://archive.ics.uci.edu/static/public/2/adult.zip && unzip \*.zip && python3 adult_generate.py && cd ..

cd diabetes && wget https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip && unzip \*.zip && python3 diabetes_generate.py && cd ..

cd census && wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gREMebIfNfuad6FAQteb4AhBNy5qyPFQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gREMebIfNfuad6FAQteb4AhBNy5qyPFQ" -O census.csv && python3 census_generate.py && cd ..

cd popsim && wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wWQ_yFuOXdk3sx-zaFGDCKT0cFFg9snH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wWQ_yFuOXdk3sx-zaFGDCKT0cFFg9snH" -O popsim_5m.csv.zip && rm -rf /tmp/cookies.txt && unzip \*.zip && python3 popsim_generate.py && cd..