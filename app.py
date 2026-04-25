import streamlit as st
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import io
import os
import urllib.request

# 페이지 설정
st.set_page_config(page_title="9-Gaze Step-by-Step", layout="centered")
st.title("👁️ 9-Gaze 정밀 AI (1장씩 릴레이 모드)")

# 1. AI 엔진 설정
@st.cache_resource
def load_ai_model():
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

detector = load_ai_model()

# 2. 세션 상태 (장바구니) 초기화
if 'photos' not in st.session_state:
    st.session_state.photos = [None] * 9
if 'step' not in st.session_state:
    st.session_state.step = 0

shooting_order = ["1. 정면", "2. 좌측", "3. 좌상측", "4. 상측", "5. 우상측", "6. 우측", "7. 우하측", "8. 하측", "9. 좌하측"]
grid_labels = ["Up-Right", "Up", "Up-Left", "Right", "Center", "Left", "Down-Right", "Down", "Down-Left"]
mapping_indices = [4, 5, 2, 1, 0, 3, 6, 7, 8] # 3x3 격자 위치

step = st.session_state.step

# 3. 스텝별 단일 업로드 UI
if step < 9:
    st.subheader(f"📍 현재 단계 ({step+1}/9): {shooting_order[step]}")
    st.info("💡 **Tip:** [Browse files] 버튼을 누르고 **'카메라'**를 선택해 바로 찍으셔도 되고, 갤러리에서 1장만 골라오셔도 됩니다.")
    
    # 단일 파일 업로더 (여러 장 선택 기능 아예 뺌)
    uploaded_file = st.file_uploader(f"{shooting_order[step]} 사진 1장 올리기", type=['jpg', 'jpeg', 'png'], key=f"uploader_{step}")
    
    if uploaded_file:
        with st.spinner("AI가 눈을 찾아 정밀 크롭 중입니다..."):
            img = Image.open(uploaded_file)
            img = ImageOps.exif_transpose(img) # 누운 사진 세우기
            img = img.convert('RGB')
            
            rgb_image = np.array(img)
            h, w = rgb_image.shape[:2]

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            res = detector.detect(mp_image)

            if res.face_landmarks:
                lm = res.face_landmarks[0]
                cx = (lm[33].x + lm[263].x) / 2 * w
                cy = (lm[33].y + lm[263].y) / 2 * h
                ew = abs(lm[263].x - lm[33].x) * w
                cw, ch = int(ew * 1.6), int(ew * 1.6 * 0.4)
                
                xmin, xmax = max(0, int(cx - cw/2)), min(w, int(cx + cw/2))
                ymin, ymax = max(0, int(cy - ch/2)), min(h, int(cy + ch/2))
                crop = rgb_image[ymin:ymax, xmin:xmax]
            else:
                st.warning("⚠️ 얼굴을 찾지 못해 임의로 자릅니다.")
                cx, cy = w // 2, h // 2
                cw, ch = int(w * 0.8), int(w * 0.8 * 0.4)
                xmin, xmax = max(0, int(cx - cw/2)), min(w, int(cx + cw/2))
                ymin, ymax = max(0, int(cy - ch/2)), min(h, int(cy + ch/2))
                crop = rgb_image[ymin:ymax, xmin:xmax]
            
            # 올바른 격자 위치에 저장하고 다음 단계로!
            target_pos = mapping_indices[step]
            st.session_state.photos[target_pos] = crop
            st.session_state.step += 1
            st.rerun()

    # 뒤로 가기 버튼
    if step > 0:
        if st.button("⬅️ 방금 올린 사진 다시 찍기"):
            st.session_state.step -= 1
            st.rerun()

# 4. 결과 출력
else:
    st.success("🎉 9방향 데이터 수집이 모두 완료되었습니다!")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    for i, label in enumerate(grid_labels):
        if st.session_state.photos[i] is not None:
            axes[divmod(i, 3)].imshow(st.session_state.photos[i])
        axes[divmod(i, 3)].set_title(label, fontsize=16, fontweight='bold')
        axes[divmod(i, 3)].axis('off')

    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("📥 논문용 결과 저장 (PNG)", buf.getvalue(), "9gaze_result.png", "image/png")
    with col2:
        if st.button("🔄 전체 새로고침 (처음부터)"):
            st.session_state.photos = [None] * 9
            st.session_state.step = 0
            st.rerun()
