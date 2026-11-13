#!/usr/bin/env bash
# Descarga los datasets públicos desde GitHub y los deja en data/raw/.
#  - rokibulroni/Phishing-Email-Dataset: Enron, SpamAssassin, Nazario, Nigerian Fraud
#    (la recopilación de Al-Subaiey et al., 2024)
#  - cramosme/LLM-Phishing-Detection: correos de phishing y legítimos generados con LLM
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw /tmp/shield_src
cd /tmp/shield_src

if [ ! -d phish ]; then
  git clone -q --filter=blob:none --no-checkout --depth 1 https://github.com/rokibulroni/Phishing-Email-Dataset phish
fi
git -C phish checkout -q HEAD -- Enron.csv SpamAssasin.csv Nazario.csv Nigerian_Fraud.csv

if [ ! -d llm ]; then
  git clone -q --filter=blob:none --no-checkout --depth 1 https://github.com/cramosme/LLM-Phishing-Detection llm
fi
git -C llm checkout -q HEAD -- data/llm-generated/legit-fixed.csv data/llm-generated/phishing-fixed.csv

cd - >/dev/null
cp /tmp/shield_src/phish/{Enron,SpamAssasin,Nazario,Nigerian_Fraud}.csv data/raw/
cp /tmp/shield_src/llm/data/llm-generated/legit-fixed.csv data/raw/llm_legit.csv
cp /tmp/shield_src/llm/data/llm-generated/phishing-fixed.csv data/raw/llm_phishing.csv
ls -la data/raw
