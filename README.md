# PREHISTORY

## How this package was created

```bash
/usr/sbin/softwareupdate --install-rosetta --agree-to-license

git clone git@github.com:schtunt/rh.git
cd rh/

pipx install virtualenv
python3 -m venv .venv

source .venv/bin/activate
python3 -m venv --upgrade .venv
python3 -m pip install --upgrade pip pipreqs

python3 -m pip install pip-autoremove pytest-mock better_exceptions rich colorhash colored_traceback ipython click python-dateutil cachier requests-cache beautifultable termcolor numpy pandas iexfinance yahoo-earnings-calendar finnhub-python robin-stocks CProfileV RunSnakeRun string-color scipy PyPortfolioOpt progress pandas_datareader yfinance tiingo alpha_vantage pyarrow fastparquet matplot

...
$ pip-autoremove --leaves
...


```
$ ./rh
```
