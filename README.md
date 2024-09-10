# Efficient time stepping for numerical integration using reinforcement learning
# Funding: NSF DMS-2110774
# https://www.nsf.gov/awardsearch/showAward?AWD_ID=2110774&HistoricalAwards=false

**
We learn step-size controllers for (1) adaptive quadrature rules and (2) adaptive integration schemes for ODEs.
Please refer to the Jupyter notebook files "quadrature.ipynb" and "time_stepper_ODE.ipynb".

These files accompany the following preprint:

    Efficient time stepping for numerical integration using reinforcement learning
    Michael Dellnitz, Eyke Hüllermeier, Marvin Lücke, Sina Ober-Blöbaum, Christian Offen, Sebastian Peitz, Karlson Pfannschmidt
    https://arxiv.org/abs/2104.03562
  

# Installing Python and Setting Up Virtual Environment

## For Ubuntu:

### Step 1: Update Package Index and Upgrade Packages

```bash
sudo apt update
sudo apt upgrade
```
### Step 2: Install Required Dependencies
```bash
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
```
### Step 3: Download Python Source Code
```bash
wget https://www.python.org/ftp/python/3.7.12/Python-3.7.12.tgz
```
### Step 4: Extract Source Code and Navigate to Directory
```bash
tar -xzvf Python-3.7.12.tgz
cd Python-3.7.12
```
### Step 5: Configure and Install Python
```bash
./configure --enable-optimizations
make
sudo make install
```
### Step 6: Verify Python Installation
```bash
python3.7 --version
```
### Step 7: Set Up Virtual Environment
```bash
python3.7 -m venv venv
source venv/bin/activate
```
## For Windows:
Follow similar steps using the Command Prompt or PowerShell on Windows. You can download Python installer from the Python official website and install it. Then, open Command Prompt or PowerShell to execute Python commands.

## For macOS:
### Step 1: Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
### Step 2: Install Python Using Homebrew
```bash
brew install python@3.7
```
### Step 3: Verify Python Installation
```bash
python3.7 --version
```
### Step 4: Set Up Virtual Environment
```bash
python3.7 -m venv venv
```
### Step 5: Activate Virtual Environment
```bash
source venv/bin/activate
```
