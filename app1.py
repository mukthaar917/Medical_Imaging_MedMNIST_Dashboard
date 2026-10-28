import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import open_clip

# 1. Define the custom classifier class (Must match your training code exactly)
class DiagnosticClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=14):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, features):
        return self.classifier(features)

# 2. Page Setup
st.title("BiomedCLIP Medical AI")
st.write("Upload a Chest X-ray to screen for 14 conditions.")

# 3. Load Models (using st.cache_resource to prevent reloading on every click)
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load BiomedCLIP
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    base_model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
    base_model = base_model.to(device)
    base_model.eval()
    
    # Load your custom trained weights
    diagnostic_head = DiagnosticClassifier().to(device)
    diagnostic_head.load_state_dict(torch.load('biomedclip_chest_classifier.pth', map_location=device))
    diagnostic_head.eval()
    
    return base_model, preprocess_val, diagnostic_head, device

base_model, preprocess_val, diagnostic_head, device = load_models()

conditions = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
              'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
              'Fibrosis', 'Pleural Thickening', 'Hernia']

# 4. UI for Upload and Prediction
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Patient X-Ray", width=300)

    if st.button("Run Diagnosis"):
        st.write("AI is analyzing the scan...")
        
        with torch.no_grad():
            # Process image and extract features
            input_tensor = preprocess_val(image).unsqueeze(0).to(device)
            features = base_model.encode_image(input_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            
            # Predict
            logits = diagnostic_head(features)
            probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Display Results
        st.subheader("Diagnostic Predictions:")
        for i, prob in enumerate(probabilities):
            status = "⚠️ DETECTED" if prob > 0.20 else "✅ Clear"
            st.text(f"{status.ljust(12)} | {conditions[i].ljust(18)} : {prob*100:05.2f}%")
            
        st.success("Analysis Complete!")