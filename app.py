import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import cv2
from openai import OpenAI
from predict import main
from resnet50 import DeepLabV3Predictor
import os
import tempfile


# --- SETUP ---
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("/Users/khemikadeedaungphan/Desktop/leaf-diseases-detect-main/sheet-430509-7be54b327849.json", scope)
client = gspread.authorize(creds)
sheet = client.open("data").sheet1

clients = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key="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",
)
model = "openai/gpt-4.1"

# Load models
modelleaf = YOLO("best-2-aug.pt")

# Initialize DeepLabV3 predictor for disease detection
MODEL_PATH = '/Users/khemikadeedaungphan/Downloads/116-cbsd-model.pt'
config = {
    'img_size': 512,
    'confidence_threshold': 0.7,
    'min_area_threshold': 100,
    'save_low_confidence': False,
    'overlay_alpha': 0.4,
    'device': 'cpu'
}
disease_predictor = DeepLabV3Predictor(MODEL_PATH, config)

st.title("โรคใบมันสำปะหลัง")

uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    st.image(image_rgb, caption="ภาพต้นฉบับ", use_container_width=True)

    # Step 1: Detect leaf using YOLO
    results = modelleaf(image_bgr)[0]
    if results.masks is None:
        st.warning("ไม่พบใบ")
        st.stop()

    masks = results.masks.data.cpu().numpy()
    mask = np.any(masks, axis=0).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    img_crop = image_bgr[y:y+h, x:x+w]
    mask_crop = mask[y:y+h, x:x+w]
    white_bg = np.full_like(img_crop, (0, 0, 0))
    mask_3c = cv2.merge([mask_crop, mask_crop, mask_crop]) // 255
    leaf_only = img_crop * mask_3c + white_bg * (1 - mask_3c)
    leaf_only = leaf_only.astype(np.uint8)

    leaf_only_rgb = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2RGB)
    st.image(leaf_only_rgb, caption="ใบมันสำปะหลังที่ตัดเฉพาะ", use_container_width=True)

    # Step 2: Use DeepLabV3 for disease detection
    # Save cropped leaf to temporary file for DeepLabV3 predictor
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        temp_path = tmp_file.name
        cv2.imwrite(temp_path, leaf_only)
    
    try:
        # Get prediction with confidence
        disease_image, disease_mask, confidence_map = disease_predictor.predict_with_confidence(temp_path)
        
        # Evaluate prediction quality
        quality_info = disease_predictor.evaluate_prediction_quality(disease_mask, confidence_map, class_id=1)
        
        if quality_info['meets_threshold'] and quality_info['area'] > 0:
            # Create overlay visualization
            overlay = disease_predictor.create_overlay(disease_image, disease_mask, confidence_map)
            st.image(overlay, caption="ตรวจพบโรค CBSD", use_container_width=True)
            
            class_name = "cbsd"
            confidence = quality_info['avg_confidence']
            st.success(f"พบ: **{class_name.upper()}** ({confidence*100:.2f}%)")
            st.info(f"พื้นที่แผล: {quality_info['area']} พิกเซล")
            
            # Prepare masks for spread analysis
            if class_name in ["cmd","cbsd","cbb","cgm"]:
                # ใช้ mask ที่ใช้ครอปตัดใบ (แปลงเป็น binary 0,1)
                mask_leaf_binary = (mask_crop > 0).astype(np.uint8)
                
                # Disease mask จาก DeepLabV3 (แปลงเป็น binary 0,1)
                disease_mask_binary = (disease_mask == 1).astype(np.uint8)
                
                main(mask_leaf_binary, disease_mask_binary)
        
        else:
            class_name = "ไม่พบแผล"
            confidence = 0
            if quality_info['reason'] == 'low_confidence':
                st.warning(f"ตรวจพบแผลแต่ความมั่นใจต่ำ ({quality_info['avg_confidence']*100:.1f}%)")
            elif quality_info['reason'] == 'small_area':
                st.warning(f"ตรวจพบแผลแต่พื้นที่เล็กเกินไป ({quality_info['area']} พิกเซล)")
            elif quality_info['reason'] == 'no_detection':
                st.warning("ไม่พบแผลในใบ")
            else:
                st.warning("ไม่พบแผลในใบ")
        
        # Clean up temporary file
        os.unlink(temp_path)
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการตรวจจับโรค: {str(e)}")
        class_name = "เกิดข้อผิดพลาด"
        confidence = 0
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    # AI recommendation
    if class_name not in ["ไม่พบแผล", "เกิดข้อผิดพลาด"]:
        prompt = f"วิเคราะห์วิธีรับมือกับต้นมันสำปะหลังที่มีโอกาสเป็นโรค {class_name} ({confidence*100:.2f}%) โดยไม่ต้องถอนต้นออก และไม่ต้องบอกให้ขอข้อมูลเพิ่มเติม"
        response = clients.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            top_p=1.0
        )
        st.subheader("ข้อแนะนำจากผู้ช่วย AI")
        st.write(response.choices[0].message.content)

    # Log to Google Sheets
    now = datetime.now().strftime("%m-%d %H:%M:%S")
    sheet.append_row([now, uploaded_file.name, class_name])

st.markdown("""
  <p style='text-align: center; font-size:10px; margin-top: 32px'>
    Chef @2025
  </p>
""", unsafe_allow_html=True)