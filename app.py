import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import time

# 페이지 설정
st.set_page_config(page_title="9-Gaze Clinical Pro", layout="centered")

st.title("👁️ 9-Gaze 정밀 촬영 시스템")

# --- 세션 상태 초기화 ---
if 'photos' not in st.session_state:
    st.session_state.photos = [None] * 9
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

shooting_order = ["1. 정면", "2. 좌측", "3. 좌상측", "4. 상측", "5. 우상측", "6. 우측", "7. 우하측", "8. 하측", "9. 좌하측"]
grid_labels = ["Up-Right", "Up", "Up-Left", "Right", "Center", "Left", "Down-Right", "Down", "Down-Left"]
mapping_indices = [4, 5, 2, 1, 0, 3, 6, 7, 8]

step = st.session_state.current_step

# --- 촬영 UI ---
if step < 9:
    st.subheader(f"📍 단계 ({step+1}/9): {shooting_order[step]}")
    
    # 💡 임상적 가이드: 코가 아닌 '눈'을 화면 중앙보다 약간 위에 맞추도록 유도
    st.error("⚠️ 중요: 양쪽 눈이 화면의 [중앙에서 약간 윗부분]에 오도록 위치시켜 주세요.")
    
    cam_photo = st.camera_input(f"{shooting_order[step]} 촬영", key=f"cam_step_{step}_{time.time()}")
    
    if cam_photo:
        img = Image.open(cam_photo).convert('RGB')
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # 🔥 [핵심 수정] 양안 중심 정렬 로직
        # 가로(X): 전체의 10%~90% (80% 폭 사용)
        # 세로(Y): 전체의 25%~45% 구간 (눈높이가 보통 얼굴의 상단에 위치함을 반영)
        # 이렇게 하면 직사각형의 가로 중심선이 코가 아닌 '눈동자'를 가로지르게 됩니다.
        left, right = int(w * 0.10), int(w * 0.90)
        top, bottom = int(h * 0.25), int(h * 0.45) 
        
        crop = img_np[top:bottom, left:right]
        
        target_pos = mapping_indices.index(step)
        st.session_state.photos[target_pos] = crop
        st.session_state.current_step += 1
        st.rerun()

    if step > 0:
        if st.button("⬅️ 이전 단계 다시 찍기"):
            st.session_state.current_step -= 1
            prev_target_pos = mapping_indices.index(st.session_state.current_step)
            st.session_state.photos[prev_target_pos] = None
            st.rerun()

# --- 결과 출력 UI ---
else:
    st.success("🎉 촬영 완료! 결과물을 확인하세요.")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    plt.subplots_adjust(wspace=0.05, hspace=0.3)

    for i, label in enumerate(grid_labels):
        if st.session_state.photos[i] is not None:
            axes[divmod(i, 3)].imshow(st.session_state.photos[i])
        axes[divmod(i, 3)].set_title(label, fontsize=12, fontweight='bold')
        axes[divmod(i, 3)].axis('off')
    
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("📥 결과 저장 (PNG)", buf.getvalue(), "9gaze_result.png", "image/png")
    with col2:
        if st.button("🔄 전체 새로고침 (데이터 삭제)"):
            st.session_state.photos = [None] * 9
            st.session_state.current_step = 0
            st.rerun()
