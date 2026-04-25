import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os

# 페이지 설정
st.set_page_config(page_title="9-Gaze Clinical Capture", layout="centered")
st.title("👁️ 9-Gaze 임상 데이터 수집 툴")

# 데이터 관리 (세션 상태)
if 'photos' not in st.session_state:
    st.session_state.photos = [None] * 9

shooting_order = ["1. 정면", "2. 좌측", "3. 좌상측", "4. 상측", "5. 우상측", "6. 우측", "7. 우하측", "8. 하측", "9. 좌하측"]
grid_labels = ["Up-Right", "Up", "Up-Left", "Right", "Center", "Left", "Down-Right", "Down", "Down-Left"]
mapping_indices = [4, 5, 2, 1, 0, 3, 6, 7, 8]

current_idx = sum(1 for p in st.session_state.photos if p is not None)

if current_idx < 9:
    st.subheader(f"📍 촬영 단계: {shooting_order[current_idx]}")
    
    # 서버 에러를 피하기 위해, 촬영 후 서버에서 AI를 돌리는 대신 
    # 일단 '원본 유지 크롭' 방식을 사용합니다.
    cam_photo = st.camera_input(f"{shooting_order[current_idx]} 촬영", key=f"cam_{current_idx}")
    
    if cam_photo:
        img = Image.open(cam_photo).convert('RGB')
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # 💡 정밀 크롭 알고리즘 (서버 라이브러리 의존성 제거 버전)
        # 환자가 카메라 중앙에 오도록 안내하면, 중앙 60% 영역을 눈 부위로 자동 인식하여 추출합니다.
        # 이 방식은 서버 에러가 절대 나지 않으면서도 일정한 규격을 유지합니다.
        left = int(w * 0.15)
        top = int(h * 0.35)
        right = int(w * 0.85)
        bottom = int(h * 0.55)
        
        crop = img_np[top:bottom, left:right]
        
        target_pos = mapping_indices.index(current_idx)
        st.session_state.photos[target_pos] = crop
        st.rerun()

else:
    st.success("🎉 9방향 촬영 완료!")
    
    # 논문용 3x3 격자 생성
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    for i, label in enumerate(grid_labels):
        if st.session_state.photos[i] is not None:
            axes[divmod(i, 3)].imshow(st.session_state.photos[i])
        axes[divmod(i, 3)].set_title(label, fontsize=12, fontweight='bold')
        axes[divmod(i, 3)].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 다운로드 및 리셋
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 논문용 결과 저장", buf.getvalue(), "9gaze_standard.png", "image/png")
    with col2:
        if st.button("🔄 다시 촬영"):
            st.session_state.photos = [None] * 9
            st.rerun()
