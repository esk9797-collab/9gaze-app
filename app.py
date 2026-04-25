import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import io
import urllib.request
import os

# 페이지 설정
st.set_page_config(page_title="9-Gaze Instant Capture", layout="centered")
st.title("📸 9-Gaze 즉석 촬영 모드")

# 세션 상태 초기화 (촬영된 사진 저장용)
if 'photos' not in st.session_state:
    st.session_state.photos = [None] * 9

# 촬영 순서와 격자 매핑
shooting_order = [
    "1. 정면", "2. 좌측", "3. 좌상측", "4. 상측", 
    "5. 우상측", "6. 우측", "7. 우하측", "8. 하측", "9. 좌하측"
]
grid_labels = [
    "Up-Right", "Up", "Up-Left",
    "Right", "Center", "Left",
    "Down-Right", "Down", "Down-Left"
]
# 촬영 순서(0~8) -> 3x3 격자 인덱스(0~8) 매핑
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

# --- 촬영 UI ---
current_idx = sum(1 for p in st.session_state.photos if p is not None)

if current_idx < 9:
    st.subheader(f"📍 현재 단계: {shooting_order[current_idx]}")
    st.write("안내된 방향을 응시한 상태에서 아래 촬영 버튼을 누르세요.")
    
    # 💡 스마트폰 카메라 호출
    cam_photo = st.camera_input(f"{shooting_order[current_idx]} 촬영하기")
    
    if cam_photo:
        # 사진 처리
        file_bytes = np.asarray(bytearray(cam_photo.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # AI 크롭
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        res = detector.detect(mp_image)
        h, w = rgb_image.shape[:2]
        
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            cx, cy = (lm[33].x + lm[263].x) / 2 * w, (lm[33].y + lm[263].y) / 2 * h
            ew = abs(lm[263].x - lm[33].x) * w
            cw, ch = int(ew * 1.6), int(ew * 1.6 * 0.4)
            xmin, xmax = max(0, int(cx-cw/2)), min(w, int(cx+cw/2))
            ymin, ymax = max(0, int(cy-ch/2)), min(h, int(cy+ch/2))
            crop = rgb_image[ymin:ymax, xmin:xmax]
        else:
            crop = rgb_image[int(h*0.3):int(h*0.5), int(w*0.1):int(w*0.9)] # 실패시 대략적 눈 위치

        # 저장
        target_pos = mapping_indices.index(current_idx)
        st.session_state.photos[target_pos] = crop
        st.rerun() # 다음 단계로 새로고침

else:
    st.success("🎉 9방향 촬영이 모두 완료되었습니다!")
    
    # 3x3 결과 출력
    fig, axes = plt.subplots(3, 3, figsize=(12, 7))
    for i, label in enumerate(grid_labels):
        axes[divmod(i, 3)].imshow(st.session_state.photos[i])
        axes[divmod(i, 3)].set_title(label, fontweight='bold')
        axes[divmod(i, 3)].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

    if st.button("🔄 다시 촬영하기"):
        st.session_state.photos = [None] * 9
        st.rerun()

    # 다운로드 버튼
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    st.download_button("📥 논문용 사진 저장하기", buf.getvalue(), "9gaze_result.png", "image/png")
