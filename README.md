Medical Imaging AI Dashboard

This repository contains a medical AI diagnostic dashboard powered by BiomedCLIP and Streamlit. The model is trained on the ChestMNIST dataset to analyze chest X-rays and predict the probabilities of 14 distinct medical conditions.

Features
* AI Diagnosis:** Utilizes a frozen BiomedCLIP base model with a custom PyTorch classification head.
* Multi-Label Detection:** Screens for conditions including Pneumonia, Cardiomegaly, Effusion, Nodules, and Atelectasis.
* Interactive UI:** A Streamlit frontend that allows users to upload patient X-ray images and instantly view diagnostic predictions.

File Structure
* app.py: The Streamlit web application script that handles the user interface and model inference.
* medicalai_dashboard.py : The source code containing the PyTorch training pipeline and dataset transformations.
* biomedclip_chest_classifier.pth: The saved PyTorch model weights generated from Google Colab.
* requirements.txt: The deployment dependencies required for Streamlit Community Cloud.

Result : 

<img width="328" height="350" alt="Xary_Result" src="https://github.com/user-attachments/assets/614a7cf8-ebff-46d1-8414-86e7068f9530" />

AI Diagnostic Predictions 

✅ Clear      | Atelectasis       : 08.20%
✅ Clear      | Cardiomegaly      : 02.03%
✅ Clear      | Effusion          : 08.28%
✅ Clear      | Infiltration      : 14.29%
✅ Clear      | Mass              : 04.17%
✅ Clear      | Nodule            : 05.03%
✅ Clear      | Pneumonia         : 01.08%
✅ Clear      | Pneumothorax      : 03.70%
✅ Clear      | Consolidation     : 03.09%
✅ Clear      | Edema             : 01.59%
✅ Clear      | Emphysema         : 01.74%
✅ Clear      | Fibrosis          : 01.30%
✅ Clear      | Pleural Thickening: 02.37%
✅ Clear      | Hernia            : 00.48%

--- Actual Ground Truth (from Doctor) ---
Patient is completely healthy (No conditions).


Deployment Instructions
To host this application for free:
1. Log into [Streamlit Community Cloud](https://share.streamlit.io) and connect your GitHub account.
2. Click **Create app** and select this repository.
3. Set the Main file path to app.py.
4. Click Deploy!
