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
st.set_page_config(page_title="9-Gaze Precision AI", layout="wide")
st.title("👁️ 9-Gaze 정밀 AI 분석기 (MediaPipe)")

# 1. AI 엔진 설정 및 모델 다운로드
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

st.info("📸 **사용 방법:** 스마트폰 기본 카메라로 정해진 순서대로 9장의 사진을 찍은 후, 아래에 한 번에 업로드해주세요.\n\n**(촬영 순서: 1.정면 ➔ 2.좌 ➔ 3.좌상 ➔ 4.상 ➔ 5.우상 ➔ 6.우 ➔ 7.우하 ➔ 8.하 ➔ 9.좌하)**")

# 2. 다중 파일 업로더
uploaded_files = st.file_uploader("사진 9장 일괄 선택", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) != 9:
        st.error(f"❌ 현재 {len(uploaded_files)}장이 선택되었습니다. 정확히 9장을 업로드해주세요.")
    else:
        with st.spinner("⏳ MediaPipe AI가 눈 랜드마크를 추적하여 정밀 크롭을 진행 중입니다..."):
            # 💡 스마트폰에서 찍힌 시간 순서대로 정렬 (파일명 오름차순)
            sorted_files = sorted(uploaded_files, key=lambda x: x.name)

            # 선생님이 지정하신 촬영 순서
            shooting_order = ["Center", "Left", "Up-Left", "Up", "Up-Right", "Right", "Down-Right", "Down", "Down-Left"]
            
            # 교과서 표준 3x3 격자 출력 순서
            grid_order = ["Up-Right", "Up", "Up-Left", "Right", "Center", "Left", "Down-Right", "Down", "Down-Left"]

            mapped_images = {}

            # 3. AI 이미지 분석 및 크롭 로직 (Colab과 100% 동일)
            for i, file in enumerate(sorted_files):
                if i >= 9: break
                
                # 이미지를 읽어서 Numpy 배열로 변환
                img = Image.open(file).convert('RGB')
                rgb_image = np.array(img)
                h, w = rgb_image.shape[:2]

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                res = detector.detect(mp_image)

                if res.face_landmarks:
                    lm = res.face_landmarks[0]
                    # 💡 논문용 정밀 고정 크롭 (양 눈끝 33번, 263번 랜드마크 기준)
                    cx = (lm[33].x + lm[263].x) / 2 * w
                    cy = (lm[33].y + lm[263].y) / 2 * h
                    ew = abs(lm[263].x - lm[33].x) * w
                    cw, ch = int(ew * 1.6), int(ew * 1.6 * 0.4)
                    
                    xmin, xmax = max(0, int(cx - cw/2)), min(w, int(cx + cw/2))
                    ymin, ymax = max(0, int(cy - ch/2)), min(h, int(cy + ch/2))

                    crop = rgb_image[ymin:ymax, xmin:xmax]
                else:
                    st.warning(f"⚠️ '{file.name}'에서 눈을 찾지 못해 임의의 비율로 자릅니다.")
                    cx, cy = w // 2, h // 2
                    cw, ch = int(w * 0.8), int(w * 0.8 * 0.4)
                    xmin, xmax = max(0, int(cx - cw/2)), min(w, int(cx + cw/2))
                    ymin, ymax = max(0, int(cy - ch/2)), min(h, int(cy + ch/2))
                    crop = rgb_image[ymin:ymax, xmin:xmax]

                current_pos = shooting_order[i]
                mapped_images[current_pos] = crop

            st.success("🎉 분석 완료! 최종 9-Gaze 논문용 격자입니다.")

            # 4. 3x3 격자 출력
            fig, axes = plt.subplots(3, 3, figsize=(12, 7))
            plt.subplots_adjust(wspace=0.1, hspace=0.3)
            
            for i, pos in enumerate(grid_order):
                axes[divmod(i, 3)].imshow(mapped_images[pos])
                axes[divmod(i, 3)].set_title(pos, fontsize=14, fontweight='bold')
                axes[divmod(i, 3)].axis('off')

            st.pyplot(fig)

            # 다운로드 버튼
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("📥 논문용 사진 저장 (PNG)", buf.getvalue(), "9gaze_result.png", "image/png")
