# if your env is replit, then use install-pke, else use sudo apt install instead

install-pkg 2>&1 > /dev/null
if [[ $? -eq 0 ]]
then
  echo "Replit env detected!"
  install-pkg libpq-dev libgl1-mesa-glx
else
  sudo apt install -y libpq-dev libgl1-mesa-glx
fi

pip install -r requirements.txt --no-cache-dir

python app.py