# Intelligent Recognition of Marine Life – Bachelor's Thesis

This repository contains a demo application for marine species recognition using deep‑learning models (**ResNet50**, **DenseNet121**, **InceptionV3**) and a user‑friendly **Streamlit** interface.

---

## How to Run the Application using Visual Studio Code

1. **Install Python**

   Make sure you have Python 3.8.

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   # Linux/Mac
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**

   ```bash
   streamlit run app.py
   ```

---

## How to Train the Models Using Anaconda + Spyder

Follow these steps if you prefer developing and training inside the Anaconda ecosystem.

### 1. Install Anaconda

Download and install Anaconda from the official page:  
<https://www.anaconda.com/products/distribution>

### 2. Create a New Environment

Open **Anaconda Prompt** (or your terminal) and create an isolated environment:

```bash
conda create -n DL-CUDA python=3.8
conda activate marine-ai
```

### 3. Install Required Libraries

With the environment active, install all necessary packages:

```bash
conda install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv
```

> *Alternative:* use pip  
> ```bash
> pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python
> ```

### 4. Install & Launch Spyder

If Spyder IDE is not already present:

```bash
conda install spyder
```

Then start it:

```bash
spyder
```

### 5. Train the Model

1. In **Spyder**, select one of the models.  
2. Hit **Run** in Spyder to start training.  
3. Monitor the console for metrics, losses, and any early‑stopping callbacks.

