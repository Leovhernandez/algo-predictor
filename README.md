Algo-Predictor Project
Overview
This project is a Python-based application (starting with a file called app.py) that uses libraries to interact with financial APIs (like Kalshi) and perform data analysis. It requires setting up Python, a code editor, and specific libraries. This README is written for complete beginners who only know that Python is a programming language. We'll break everything down into basic truths: Python code needs an "interpreter" (a program that runs your code), libraries are pre-written code you can reuse, and tools like virtual environments keep things organized without conflicts.
From first principles:

Fundamental 1: Computers don't understand Python code directly; they need a Python interpreter to translate and run it.
Fundamental 2: Projects often use extra code (libraries) from others. These must be "installed" into your Python setup.
Fundamental 3: To avoid messing up other projects, we use "virtual environments" – isolated spaces for each project's interpreter and libraries.
Fundamental 4: A code editor (like VS Code) helps write and run code, with features to spot errors early.

By building from these basics, you'll set up everything logically and reproducibly. Follow these steps exactly on your Windows machine (laptop or desktop). This replicates the setup from the original conversation.
Prerequisites
Before starting, ensure you have:

A Windows computer.
Internet access for downloads.

Step-by-Step Setup Instructions (NOTE: these instructions assume you are using a Windows machine)
Step 1: Install Python
Python is the core interpreter. We need version 3.14.2 (a specific, stable release, verified as released on December 5, 2025).

Go to the official Python website: https://www.python.org/downloads/.
Download the Windows installer for Python 3.14.2 (look for the "Windows installer (64-bit)" if your computer is 64-bit, which most are).
Run the installer file.
In the installer window:
Check the box: "Add python.exe to PATH" (this lets you run Python from anywhere).
Click "Install Now."

After installation, open Command Prompt (search for "cmd" in the Start menu).
Type python --version and press Enter. It should show "Python 3.14.2". If not, restart your computer and try again.

Step 2: Install Visual Studio Code (VS Code)
VS Code is a free code editor that understands Python and helps fix errors.

Go to: https://code.visualstudio.com/.
Download the Windows version.
Run the installer and follow the prompts (default options are fine).
Open VS Code after installation.
In VS Code, click the Extensions icon (looks like squares) on the left sidebar.
Search for "Python" (by Microsoft) and install it. This adds Python support, including error checking.

Step 3: Set Up the Project Folder

Create a new folder on your computer, e.g., C:\quant\algo-predictor.
Copy or create your code files (like app.py) into this folder.
If you have a requirements.txt file (see below), put it here too. This file lists the libraries needed.

Step 4: Create a Virtual Environment (Venv)
A venv is an isolated Python setup just for this project – like a private room to avoid cluttering the whole house.

Open Command Prompt.
Navigate to your project folder: Type cd C:\quant\algo-predictor (replace with your path) and press Enter.
Create the venv: Type python -m venv venv and press Enter. This makes a folder called "venv" inside your project.

Step 5: Activate the Virtual Environment
Activating switches to the project's isolated Python.

In the same Command Prompt: Type venv\Scripts\activate and press Enter.
Your prompt will change to show (venv) at the start – this means it's active.

Step 6: Install Required Libraries
Libraries are like toolkits: kalshi-python for API access, scipy for math/stats, pandas for data handling.

We use a requirements.txt file to list them exactly, ensuring the same versions everywhere.


If not already there, create a file called requirements.txt in your project folder with this content (open in Notepad or VS Code):textkalshi-python
scipy
pandas
With venv active in Command Prompt: Type pip install -r requirements.txt and press Enter. This installs the libraries only in the venv, not globally.
If scipy fails (it needs extra tools), download "Visual Studio Build Tools" from https://visualstudio.microsoft.com/downloads/. Install the "Desktop development with C++" option, then retry the install.

Verify: Type pip list – you should see the libraries listed.

Step 7: Open the Project in VS Code

In VS Code: Click File > Open Folder, select your project folder (e.g., C:\quant\algo-predictor).
Press Ctrl+Shift+P (or Cmd+Shift+P on Mac, but you're on Windows) to open the Command Palette.
Type "Python: Select Interpreter" and select it.
Choose the one in your venv: It looks like .\venv\Scripts\python.exe.
Reload VS Code: Ctrl+Shift+P, type "Developer: Reload Window", select it.


Now, open app.py. Any wavy lines (warnings) under imports should disappear. If not, Ctrl+Shift+P > "Python: Restart Language Server".

Step 8: Run Your Code

In VS Code, open the Terminal: Ctrl+` (backtick key).
If not already, activate venv: Type venv\Scripts\activate
NOTE: If you run into an issue where the above command throws an error like "cannot be loaded because running scripts is 
disabled on this system", do the following to fix this error:
    - Enter the following command: Get-ExecutionPolicy
        - if output says Restricted, then enter the following command: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
        - Enter the Get-ExecutionPolicy command again and the output should be "RemoteSigned" now granting you access to activate the virtual environment
        - Finally, activate venv by typing and entering the following command again: venv\Scripts\activate
        - The virtual environment will be activated if you see (venv) appear before the filepath.
Run: Type python app.py and press Enter.


If other errors occur (e.g., about imports), double-check the interpreter selection and installations.

Step 9: Deactivate and Clean Up

When done: In Command Prompt or Terminal, type deactivate to exit the venv.
Don't delete the venv folder – it's your project's setup. If syncing the project (e.g., via Git or cloud), ignore the venv folder (it's machine-specific).

Why This Setup? (From Fundamentals)

Isolation: Global installs can conflict (e.g., different projects need different library versions). Venv fixes this by starting from a clean slate per project.
Reproducibility: requirements.txt rebuilds the exact environment anywhere, based on the truth that packages have versions.
Error Prevention: VS Code's tools check code against the active environment, catching issues early.

Troubleshooting

Command not found? Check PATH (restart Command Prompt).
Still warnings? Reselect interpreter or reinstall packages.
For more help, search "Python venv Windows tutorial" online.

Now you're ready to code! Edit app.py and run it.