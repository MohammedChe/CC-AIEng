# Python installation on Windows

1. Search for 'Python Install Manager' in Microsoft Store
2. Run the Install Manager

- Open Installed apps now? - select N
- Add commands directory to your PATH - select Y
- Install CPython now - select Y
- Select no for viewing online
- Close powershell/terminal
- Open Powershell
- run `python -V`
- it should show v3.14

3. Open this repo in VSCode and run the following in a Powershell terminal

- delete the .venv folder if it exists (restart terminal)
- python -m venv ./.venv
- Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
- . .\.venv\Scripts\Activate.ps1
- python -m pip install --upgrade pip
- python -m pip install -r ./requirements.txt

# Instructions for Mac

1. Install Python 3.14

- Check the version using `python -V` or `python3 -V`
- use python or python3 for the rest of the commands (whichever displayed 3.14)

2. Open this repo in VSCode and run the following in the terminal

- delete the .venv folder if it exists (restart terminal)
- python3 -m venv ./.venv
- source ./.venv/bin/activate
- after the above command 'python -V' should work if you were using 'python3'
- python -m pip install --upgrade pip
- python -m pip install -r ./requirements.txt
