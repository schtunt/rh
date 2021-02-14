# PREHISTORY

## How this package was created

```bash
git clone git@github.com:schtunt/rh.git
cd rh/

echo .venv > .gitignore

pipx install virtualenv
python3 -m venv --upgrade-deps --upgrade .venv
python3 -m pip install --upgrade pip pipreqs
source .venv/bin/activate

pip install robin-stocks
pip install numpy
pip install python-dateutil
pip install polygon-api-client --include-deps
pip install iexfinance

...

$ pip-autoremove --leaves
yahoo-earnings-calendar 0.6.0 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
termcolor 1.1.0 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
tabulate 0.8.7 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
RunSnakeRun 2.0.5 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
robin-stocks 1.7.0 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
requests-cache 0.5.2 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
pytest-mock 3.5.1 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
pyclick 0.0.2 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
prettytable 2.0.0 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
polygon-api-client 0.1.9 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
pipreqs 0.4.10 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
pip 21.0.1 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
pip-autoremove 0.9.1 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
monkeylearn 3.5.2 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
isort 5.7.0 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
ipython 7.19.0 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
install 1.3.4 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
iexfinance 0.5.0 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
finnhub-python 2.4.0 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
CProfileV 1.0.7 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
click 7.1.2 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
cachier 1.5.0 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
beautifultable 1.0.1 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)
beautifulsoup4 4.9.3 (/Users/nimbler/rh/.venv/lib/python3.9/site-packages)

...

deactivate

```

# DEMO
```
$ touch ~/.rhrc
$ chmod 0600 ~/.rhrc
$ echo "${RH_USERNAME?},${RH_PASSWORD} > ~/.rhrc
```

```
$ ./rh
```
