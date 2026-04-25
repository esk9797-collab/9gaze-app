import streamlit as st
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from PIL import Image
import io
import urllib.request
import os

# 페이지 설정
st.set_page_config(page_title="9-Gaze Instant Capture", layout="centered")
st.title("📸 9-Gaze 즉석 촬영 모드")

# 세션 상태 초기화
if 'photos' not in st.session_state:
    st.session_state.photos = [None] * 9

shooting_order = ["1. 정면", "2. 좌측", "3. 좌상측", "4. 상측", "5. 우상측", "6. 우측", "7. 우하측", "8. 하측", "9. 좌하측"]
grid_labels = ["Up-Right", "Up", "Up-Left", "Right", "Center", "Left", "Down-Right", "Down", "Down-Left"]
mapping_indices = [4, 5, 2, 1, 0, 3, 6, 7, 8]

# AI 모델 로드
@st.cache_resource
def load_model():
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

detector = load_model()

# 현재 촬영 단계 확인
current_idx = sum(1 for p in st.session_state.photos if p is not None)

if current_idx < 9:
    st.subheader(f"📍 현재 단계: {shooting_order[current_idx]}")
    cam_photo = st.camera_input(f"{shooting_order[current_idx]} 촬영하기")
    
    if cam_photo:
        # PIL을 사용하여 이미지 읽기 (OpenCV 미사용)
        img = Image.open(cam_photo).convert('RGB')
        img_np = np.array(img)
        
        # AI 크롭 로직
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        res = detector.detect(mp_image)
        h, w = img_np.shape[:2]
        
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            cx, cy = (lm[33].x + lm[263].x) / 2 * w, (lm[33].y + lm[263].y) / 2 * h
            ew = abs(lm[263].x - lm[33].x) * w
            cw, ch = int(ew * 1.6), int(ew * 1.6 * 0.4)
            xmin, xmax = max(0, int(cx-cw/2)), min(w, int(cx+cw/2))
            ymin, ymax = max(0, int(cy-ch/2)), min(h, int(cy+ch/2))
            crop = img_np[ymin:ymax, xmin:xmax]
        else:
            # 인식 실패 시 대략적인 눈 위치 크롭
            crop = img_np[int(h*0.35):int(h*0.55), int(w*0.1):int(w*0.9)]

        # 저장 및 리프레시
        target_pos = mapping_indices.index(current_idx)
        st.session_state.photos[target_pos] = crop
        st.rerun()

else:
    st.success("🎉 촬영 완료!")
    fig, axes = plt.subplots(3, 3, figsize=(12, 7))
    for i, label in enumerate(grid_labels):
        if st.session_state.photos[i] is not None:
            axes[divmod(i, 3)].imshow(st.session_state.photos[i])
        axes[divmod(i, 3)].set_title(label, fontweight='bold')
        axes[divmod(i, 3)].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

    if st.button("🔄 다시 촬영하기"):
        st.session_state.photos = [None] * 9
        st.rerun()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    st.download_button("📥 논문용 사진 저장", buf.getvalue(), "9gaze_result.png", "image/png")
