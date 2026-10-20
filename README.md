# AI-Powered Diabetic Foot Ulcer (DFU) Classification System

An AI-powered application for detecting and classifying diabetic foot conditions using deep learning and computer vision. The project aims to assist in the identification of diabetic foot ulcers and support preliminary assessment through an interactive web interface.

## 📌 Project Overview

Diabetic Foot Ulcers (DFUs) are among the serious complications associated with diabetes. Early identification and appropriate clinical assessment can help reduce the risk of complications.

This project explores the use of deep learning to classify foot images and provide prediction results through a web-based application developed using Streamlit.

The system is designed as a prototype for educational and research purposes and is not intended to replace professional medical diagnosis.

## 🎯 Objectives

- Develop an AI-based system for diabetic foot condition classification.
- Apply deep learning techniques to analyze foot images.
- Provide an easy-to-use interface for uploading images and viewing predictions.
- Support preliminary assessment through automated classification.
- Explore model evaluation and visualization using Jupyter notebooks.
- Generate patient-oriented reports where supported by the application.

## ✨ Key Features

- **Image-based classification:** Analyze uploaded foot images using a trained deep learning model.
- **Interactive web application:** Built with Streamlit.
- **Model integration:** Supports trained models compatible with the application's inference pipeline.
- **Evaluation notebooks:** Includes notebooks for model training, testing, and performance analysis.
- **Report generation:** Includes a notebook for PDF report generation.
- **Modular project structure:** Separates application code, source modules, notebooks, assets, and supporting files.

*The availability of individual features depends on the models and modules configured in the application.*

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Pillow
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 📂 Project Structure

```text
DFU_Classification_Project/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   └── Application assets
│
├── data/
│   └── Dataset files (not included in Git)
│
├── models/
│   └── Trained model files (may be hosted separately)
│
├── notebooks/
│   ├── DFU_Detection_Model.ipynb
│   ├── DFU_Heatmap_Evaluation.ipynb
│   ├── DFU_Patient_Report_Generation.ipynb
│   └── DFU_PDF_Report_Generation.ipynb
│
├── outputs/
│   └── Generated outputs
│
└── src/
    └── Supporting source code
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/DFU-Classification-Project.git
cd DFU-Classification-Project
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

For Windows:

```powershell
py -3.10 -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure the trained model

Place the trained model files in the location expected by `app.py`.

If model files are hosted separately, download them and update the model paths in the application configuration as necessary.

**Note:** The repository must include, or provide instructions for obtaining, all required model files before inference can work.

### 6. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## 🔬 Model Development and Evaluation

The project includes Jupyter notebooks for model development and evaluation.

Typical activities include:

- Dataset preparation and preprocessing.
- Training and validation of deep learning models.
- Evaluation on held-out test data.
- Analysis of classification performance.
- Visualization of model predictions and heatmaps, where implemented.
- Generation of assessment reports, where supported.

The actual architecture, class labels, preprocessing steps, and evaluation metrics should be documented according to the final trained model and experimental results.

## 📊 Performance Evaluation

Model performance should be evaluated using appropriate metrics, including:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-score
- Confusion matrix

For medical-image classification, accuracy alone is not sufficient. Recall, precision, class-wise performance, and testing on independent datasets are also important.

Add your final verified test results and confusion matrix here after confirming the evaluation methodology.

## 📋 Requirements

- Python 3.10
- A compatible operating system and Python environment
- Dependencies listed in `requirements.txt`
- Trained model files required by the application
- Sufficient memory and processing capacity for model inference

GPU support is optional and depends on the installed machine-learning framework and hardware configuration.

## ⚠️ Medical Disclaimer

This project is intended for academic, educational, and research purposes only.

It is a prototype and has not been established as a clinically validated diagnostic system. Predictions may be incorrect, and the application must not be used as a substitute for professional medical evaluation, diagnosis, or treatment. Clinical decisions should be made by qualified healthcare professionals.

## 🔒 Data Privacy

- Do not upload identifiable patient images or medical records to a public repository.
- Ensure that datasets are used in accordance with their licenses and applicable privacy requirements.
- Do not commit passwords, API keys, access tokens, or other confidential information.
- Keep large datasets and sensitive generated reports outside the public repository.

## 🚀 Future Improvements

- Improve classification performance across different ulcer grades and foot conditions.
- Evaluate generalization on independent datasets.
- Enhance explainability using suitable visualization techniques.
- Improve the user interface and report-generation workflow.
- Validate the system with appropriately designed clinical studies.
- Optimize model inference for deployment on resource-constrained devices.

## 👨‍💻 Author

**Adhil Nazeer**

B.Tech – Electronics and Communication Engineering

GitHub: [YOUR-GITHUB-USERNAME](https://github.com/YOUR-USERNAME)

## 📄 License

A license has not yet been specified. Add an appropriate license after deciding how the code may be reused and distributed.

## 🙏 Acknowledgements

Thanks to the researchers and dataset providers whose publicly available resources support research into diabetic foot ulcer detection and classification. All datasets and external resources should be credited according to their respective licenses and citation requirements.
