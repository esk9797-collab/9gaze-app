import streamlit as st
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import urllib.request

# 페이지 설정
st.set_page_config(page_title="9-Gaze Precision Capture", layout="centered")
st.title("👁️ 9-Gaze 정밀 진단 모드")

# 모델 파일 강제 다운로드 함수
@st.cache_resource
def get_model_file():
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
    return model_path

# AI 모델 로드 (정밀 모드)
@st.cache_resource
def load_precision_model(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

# 모델 초기화
try:
    model_file = get_model_file()
    detector = load_precision_model(model_file)
    st.success("✅ AI 정밀 진단 엔진이 가동 중입니다.")
except Exception as e:
    st.error(f"❌ 엔진 가동 실패: {e}")
    st.info("관리자에게 시스템 라이브러리(libgl1 등) 확인을 요청하세요.")
    st.stop()

# --- 데이터 관리 ---
if 'photos' not in st.session_state:
    st.session_state.photos = [None] * 9

shooting_order = ["1. 정면", "2. 좌측", "3. 좌상측", "4. 상측", "5. 우상측", "6. 우측", "7. 우하측", "8. 하측", "9. 좌하측"]
grid_labels = ["Up-Right", "Up", "Up-Left", "Right", "Center", "Left", "Down-Right", "Down", "Down-Left"]
mapping_indices = [4, 5, 2, 1, 0, 3, 6, 7, 8]

current_idx = sum(1 for p in st.session_state.photos if p is not None)

if current_idx < 9:
    st.subheader(f"📍 촬영 방향: {shooting_order[current_idx]}")
    cam_photo = st.camera_input(f"{shooting_order[current_idx]} 촬영", key=f"cam_{current_idx}")
    
    if cam_photo:
        img = Image.open(cam_photo).convert('RGB')
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # 정밀 크롭 프로세스 (Mediapipe 기반)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        res = detector.detect(mp_image)
        
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            # 눈동자 중심 좌표 계산
            cx = (lm[33].x + lm[263].x) / 2 * w
            cy = (lm[33].y + lm[263].y) / 2 * h
            # 양안 거리 기반 크롭 사이즈 결정 (정규화)
            eye_dist = abs(lm[263].x - lm[33].x) * w
            cw, ch = int(eye_dist * 1.8), int(eye_dist * 1.8 * 0.45)
            
            xmin, xmax = max(0, int(cx-cw/2)), min(w, int(cx+cw/2))
            ymin, ymax = max(0, int(cy-ch/2)), min(h, int(cy+ch/2))
            crop = img_np[ymin:ymax, xmin:xmax]
            
            target_pos = mapping_indices.index(current_idx)
            st.session_state.photos[target_pos] = crop
            st.rerun()
        else:
            st.warning("⚠️ 얼굴이 인식되지 않았습니다. 다시 밝은 곳에서 찍어주세요.")

else:
    # (결과 출력 부분 동일)
    st.success("🎉 모든 데이터 수집 완료!")
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    for i, label in enumerate(grid_labels):
        axes[divmod(i, 3)].imshow(st.session_state.photos[i])
        axes[divmod(i, 3)].set_title(label, fontsize=12, fontweight='bold')
        axes[divmod(i, 3)].axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    st.download_button("📥 논문용 데이터 저장", buf.getvalue(), "9gaze_precision.png", "image/png")
    if st.button("🔄 다시 시작"):
        st.session_state.photos = [None] * 9
        st.rerun()
        st.download_button("📥 논문용 사진 저장(PNG)", buf.getvalue(), "9gaze_result.png", "image/png")
