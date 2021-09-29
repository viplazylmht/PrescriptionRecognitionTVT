# if your env is replit, then use install-pke, else use sudo apt install instead

install-pkg 2>&1 > /dev/null
if [[ $? -eq 0 ]]
then
  echo "Replit env detected!"
  install-pkg libpq-dev tesseract-ocr
  export TESSDATA_PREFIX="./im2pres/data"
else
  sudo apt install libpq-dev tesseract-ocr
  sudo -E export TESSDATA_PREFIX="./im2pres/data"
fi

pip install -r requirements.txt --no-cache-dir

python app.py