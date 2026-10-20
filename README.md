# AI-Powered Diabetic Foot Ulcer (DFU) Classification System

An AI-powered application for detecting and classifying diabetic foot conditions using deep learning and computer vision. This project aims to support preliminary assessment through an interactive web application built with Streamlit.

## 📌 Project Overview

Diabetic Foot Ulcers (DFUs) are serious complications associated with diabetes. Early identification and appropriate clinical assessment are important for reducing the risk of complications.

This project explores deep learning techniques for analyzing foot images and providing classification predictions through a web-based application.

The system is developed as an academic and research prototype. It is not intended to replace professional medical diagnosis.

## 🎯 Objectives

- Develop an AI-based system for diabetic foot condition classification.
- Apply deep learning to analyze foot images.
- Provide a user-friendly interface for uploading images and viewing predictions.
- Explore model evaluation and visualization techniques.
- Support report generation where implemented in the application.
- Organize the project into reusable modules and notebooks.

## ✨ Key Features

- **Image classification:** Analyze uploaded foot images using a trained deep learning model.
- **Interactive interface:** Web application developed with Streamlit.
- **Deep learning integration:** Uses trained models compatible with the inference pipeline.
- **Model evaluation:** Jupyter notebooks for testing and performance analysis.
- **Heatmap visualization:** Includes a notebook for heatmap evaluation.
- **Report generation:** Includes notebooks for patient-oriented PDF report generation.
- **Modular structure:** Separates application code, notebooks, model files, and supporting resources.

Feature availability depends on the model files and modules configured in the application.

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Jupyter Notebook

## 📂 Project Structure

```text
DFU-Classification-Project/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── assets/
│
├── data/
│   └── Dataset files (excluded from Git)
│
├── models/
│   └── Trained model files (excluded from Git)
│
├── notebooks/
│   ├── DFU_Detection_Model.ipynb
│   ├── DFU_Heatmap_Evaluation.ipynb
│   ├── DFU_PDF_Report_Generation.ipynb
│   └── DFU_Patient_Report_Generation.ipynb
│
├── outputs/
│   └── Generated evaluation results and reports
│
└── src/
    └── Supporting source code
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/AdhilNazeer/DFU-Classification-Project.git
cd DFU-Classification-Project
```

### 2. Create a virtual environment

For Windows:

```powershell
py -3.10 -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure the trained model

The trained model files are not included in this GitHub repository because large model files are excluded by `.gitignore`.

Before running the application, obtain the required trained model files and place them in the locations expected by `app.py`.

Update the model paths in the application if your model files are stored elsewhere.

### 6. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually:

http://localhost:8501

## 🔬 Model Development and Evaluation

The project includes Jupyter notebooks for model development, evaluation, heatmap analysis, and report generation.

Typical activities include:

- Dataset preparation and preprocessing.
- Training and validation of deep learning models.
- Testing on held-out images.
- Analysis of classification performance.
- Confusion matrix generation.
- Heatmap visualization where implemented.
- Patient-oriented report generation where supported.

The exact model architecture, preprocessing pipeline, class labels, and evaluation methodology should be verified against the implementation in the notebooks and application.

## 📊 Evaluation Metrics

The model can be evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-score
- Confusion matrix

For medical image classification, accuracy alone is insufficient. Class-wise performance, recall, precision, and evaluation on independent datasets are also important.

Add final verified test results and the corresponding confusion matrix here when available. Training accuracy should not be presented as test accuracy.

## 📋 Requirements

- Python 3.10
- Dependencies listed in `requirements.txt`
- Trained model files required by the application
- Sufficient memory for model inference

GPU support is optional and depends on the installed machine-learning framework, drivers, and hardware.

## 🔒 Data Privacy

- Do not upload identifiable patient images or medical records to a public repository.
- Use datasets according to their licenses and applicable privacy requirements.
- Do not commit API keys, passwords, access tokens, or confidential information.
- Keep sensitive generated reports and private data outside the repository.

## ⚠️ Medical Disclaimer

This project is intended for academic, educational, and research purposes only.

It is a prototype and has not been established as a clinically validated diagnostic system. Predictions may be incorrect and must not be used as a substitute for professional medical evaluation, diagnosis, or treatment. Clinical decisions should be made by qualified healthcare professionals.

## 🚀 Future Improvements

- Improve classification performance across different ulcer grades and foot conditions.
- Evaluate model generalization on independent datasets.
- Enhance explainability through appropriate visualization methods.
- Improve the user interface and reporting workflow.
- Optimize inference for resource-constrained devices.
- Evaluate the system through appropriately designed clinical studies.

## 👨‍💻 Author

**Adhil Nazeer**

B.Tech – Electronics and Communication Engineering

- **GitHub Profile:** [AdhilNazeer](https://github.com/AdhilNazeer)
- **Project Repository:** [DFU Classification Project](https://github.com/AdhilNazeer/DFU-Classification-Project)

## 📄 License

No license has been specified for this project yet. A suitable license can be added after deciding the terms under which others may use, modify, and distribute the code.

## 🙏 Acknowledgements

Acknowledgements and citations should be added for the datasets, research papers, pretrained models, and other resources used in this project. All third-party materials should be credited according to their applicable licenses and citation requirements.
