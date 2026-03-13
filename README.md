# plant-disease-yield-prediction
## STEP 1: Install Python (The Right Way)
Don't use Microsoft Store Python. Use Miniconda (recommended for data science):
### Download Miniconda:
Go to: https://docs.conda.io/en/latest/miniconda.html
Download: Miniconda3 Windows 64-bit
Install with "Add to PATH" checked
### Verify installation:
Open Anaconda Prompt
Type: conda --version
Should show something like: conda 23.x.x
## STEP 2: Create Your Project Environment
Open Anaconda Prompt and run these commands ONE BY ONE:
```
# Create environment with Python 3.9 (stable for TensorFlow)
conda create -n plant_disease python=3.9 -y

# Activate the environment
conda activate plant_disease

# Install CUDA toolkit and cuDNN (this handles ALL compatibility issues!)
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y

# Set environment variables (CRITICAL for Windows)
conda env config vars set LD_LIBRARY_PATH=%CONDA_PREFIX%\Library\bin

# Reactivate to load variables
conda deactivate
conda activate plant_disease

# Install TensorFlow (version compatible with CUDA 11.2)
pip install tensorflow==2.10

# Install other required packages
pip install pandas numpy matplotlib seaborn scikit-learn jupyter notebook opencv-python pillow
```
## STEP 3: Setup
### Open Anaconda Prompt (in your project folder):
```
# Verify GPU is working
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

# Install git if not present
conda install git -y

# Configure git (use your actual name and email)
git config --global user.name "your_username"
git config --global user.email "your.email@example.com"

# Clone your repository (replace with your actual URL)
cd C:\Users\user\Desktop\folder_name
git clone https://github.com/zahraghourani/plant-disease-yield-prediction.git

# Go into the folder
cd plant-disease-yield-prediction

# Create folder structure
mkdir data
mkdir models
mkdir notebooks
mkdir src
mkdir results
mkdir checkpoints

# Create .gitignore file (prevents uploading large files)
echo "data/
checkpoints/
*.h5
*.pkl
__pycache__/
.ipynb_checkpoints/" > .gitignore

# Add everything to git
git add .
git commit -m "Initial project structure"

# Push to GitHub
git push origin main
```
### Expected Output:
`GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

## STEP 4: Download the Dataset
### Option 1: Manual Download
#### Go to: https://www.kaggle.com/datasets/nafishamoin/bangladeshi-crops-disease-dataset
### Option 2: Kaggle API (For automation)
```
# Install Kaggle API
pip install kaggle

# Get API key from Kaggle (Profile → Account → Create New API Token)
# Place kaggle.json in C:\Users\zahra.elghourani\.kaggle\

# Download dataset
kaggle datasets download -d nafishamoin/bangladeshi-crops-disease-dataset
unzip bangladeshi-crops-disease-dataset.zip -d data/
```
## STEP 5: How to run the Keras Models(for each person)
```
conda activate plant_disease
cd "C:\path\to\project"
python src/train_keras_models.py --person _person_name_ (zahra/sireen/tala)
```
## STEP 6: Git Workflow
### Daily Workflow:
```
# 1. Get latest changes from teammates
git pull origin main

# 2. Do your work (train models, update code)

# 3. Save your changes
git add .
git commit -m "Trained ResNet50, achieved 94% accuracy"

# 4. Upload to GitHub
git push origin main
```
### If there is a conflict:
```
git pull origin main --rebase
# Fix any conflicts in files
git add .
git rebase --continue
git push origin main
```