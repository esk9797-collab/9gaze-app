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

shooting_order = [
    "1. 정면", "2. 좌측", "3. 좌상측",
    "4. 상측", "5. 우상측", "6. 우측",
    "7. 우하측", "8. 하측", "9. 좌하측"
]

grid_labels = [
    "Up-Right", "Up", "Up-Left",
    "Right", "Center", "Left",
    "Down-Right", "Down", "Down-Left"
]

mapping_indices = [4, 5, 2, 1, 0, 3, 6, 7, 8]

step = st.session_state.current_step

# --- 촬영 UI ---
if step < 9:
    st.subheader(f"📍 단계 ({step+1}/9): {shooting_order[step]}")

    st.warning("📱 전면 카메라로 전환 후 촬영하세요.")
    st.error("⚠️ 양쪽 눈이 화면의 '중앙보다 약간 위'에 오도록 맞춰주세요.")

    # ✅ key 고정 (중요)
    cam_photo = st.camera_input(
        f"{shooting_order[step]} 촬영",
        key=f"cam_step_{step}"
    )

    if cam_photo is not None:
        img = Image.open(cam_photo).convert('RGB')
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        # 눈 중심 crop
        left, right = int(w * 0.10), int(w * 0.90)
        top, bottom = int(h * 0.25), int(h * 0.45)

        crop = img_np[top:bottom, left:right]

        target_pos = mapping_indices.index(step)
        st.session_state.photos[target_pos] = crop

        # ✅ 안정적인 다음 단계 이동
        time.sleep(0.2)
        st.session_state.current_step += 1
        st.rerun()

    # 이전 단계 버튼
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
        ax = axes[i // 3, i % 3]

        if st.session_state.photos[i] is not None:
            ax.imshow(st.session_state.photos[i])

        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.axis('off')

    st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button(
            "📥 결과 저장 (PNG)",
            buf.getvalue(),
            "9gaze_result.png",
            "image/png"
        )

    with col2:
        if st.button("🔄 전체 새로고침 (데이터 삭제)"):
            st.session_state.photos = [None] * 9
            st.session_state.current_step = 0
            st.rerun()
