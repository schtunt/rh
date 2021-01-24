# PREHISTORY

## How this package was created

```bash
git clone git@github.com:schtunt/rh.git
cd rh/

echo .venv > .gitignore

pipx install virtualenv
python3 -m venv --upgrade-deps --upgrade .venv

python3 -m pip install --upgrade pip
python3 -m pip install robin-stocks
python3 -m pip install numpy
python3 -m pip install python-dateutil

source .venv/bin/activate

deactivate
```
