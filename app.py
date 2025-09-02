import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="OpenCV Projects", layout="wide")
st.title("🎥 OpenCV + Mediapipe Live Projects")
st.sidebar.title("Choose a Project")

project = st.sidebar.selectbox(
    "Select one",
    [
        "Hand Tracking",
        "Gesture Media Control",
        "Face Detection",
        "Pose Estimation",
        "Virtual Painter",
    ],
)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose

# Webcam
cap = cv2.VideoCapture(0)

FRAME_WINDOW = st.image([])


def hand_tracking():
    with mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, landmarks, mp_hands.HAND_CONNECTIONS
                    )

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def face_detection():
    with mp_face.FaceDetection(min_detection_confidence=0.6) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pose_estimation():
    with mp_pose.Pose(
        min_detection_confidence=0.6, min_tracking_confidence=0.6
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def virtual_painter():
    brush_color = (0, 0, 255)  # Red brush
    canvas = None
    with mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if canvas is None:
                canvas = np.zeros_like(frame)

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    h, w, _ = frame.shape
                    x = int(landmarks.landmark[8].x * w)
                    y = int(landmarks.landmark[8].y * h)
                    cv2.circle(canvas, (x, y), 8, brush_color, -1)

            frame = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


if project == "Hand Tracking":
    hand_tracking()
elif project == "Face Detection":
    face_detection()
elif project == "Pose Estimation":
    pose_estimation()
elif project == "Virtual Painter":
    virtual_painter()
elif project == "Gesture Media Control":
    st.subheader("🖐 Gesture Controlled Media Player")

    import pyautogui
    import time

    # Sidebar controls & debug placeholders
    cooldown = st.sidebar.slider("Cooldown (s)", 0.2, 2.0, 0.8, 0.1)
    show_landmarks = st.sidebar.checkbox("Show landmarks", True)
    debug_box = st.sidebar.empty()
    test_box = st.sidebar.empty()

    # Try to use pycaw for reliable volume control on Windows; fallback to pyautogui if unavailable
    use_pycaw = False
    audio_volume = None
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        audio_volume = cast(interface, POINTER(IAudioEndpointVolume))
        use_pycaw = True
    except Exception as e:
        use_pycaw = False

    def get_volume_scalar():
        if use_pycaw and audio_volume is not None:
            try:
                return float(audio_volume.GetMasterVolumeLevelScalar())
            except Exception:
                return None
        return None

    def change_volume(delta_scalar=0.05):
        """delta_scalar: positive to increase, negative to decrease (0..1 range)."""
        if use_pycaw and audio_volume is not None:
            try:
                cur = audio_volume.GetMasterVolumeLevelScalar()  # 0.0 - 1.0
                new = max(0.0, min(1.0, cur + float(delta_scalar)))
                audio_volume.SetMasterVolumeLevelScalar(new, None)
                return True
            except Exception:
                pass
        # fallback: try pyautogui multimedia keys
        try:
            if delta_scalar > 0:
                pyautogui.press("volumeup")
            else:
                pyautogui.press("volumedown")
            return True
        except Exception:
            return False

    # Sidebar test buttons
    if test_box.button("Test Volume +"):
        ok = change_volume(+0.05)
        st.sidebar.success("Test Volume +: " + ("OK" if ok else "FAILED"))
    if test_box.button("Test Next Track"):
        try:
            pyautogui.press("nexttrack")
            st.sidebar.success("Test Next Track: OK")
        except Exception:
            try:
                pyautogui.hotkey("ctrl", "right")
                st.sidebar.success("Test Next Track (fallback): OK")
            except Exception:
                st.sidebar.error("Test Next Track: FAILED")

    st.sidebar.markdown(f"pycaw active: **{use_pycaw}**")
    vol_now = get_volume_scalar()
    st.sidebar.markdown(
        f"Current master volume: **{vol_now:.2f}**"
        if vol_now is not None
        else "Current master volume: **N/A**"
    )

    last_action_time = 0.0

    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam frame not available")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            now = time.time()
            gesture_text = ""
            debug_lines = []

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    if show_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, landmarks, mp_hands.HAND_CONNECTIONS
                        )

                    lm = landmarks.landmark  # normalized coords
                    h, w, _ = frame.shape

                    # Build fingers open/closed array (thumb, index, middle, ring, pinky)
                    tip_ids = [4, 8, 12, 16, 20]
                    fingers = []
                    for tip in tip_ids:
                        # compare tip vs pip (tip-2) in y (smaller y -> higher on screen)
                        try:
                            fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)
                        except Exception:
                            fingers.append(0)

                    debug_lines.append(f"fingers: {fingers}")
                    # index tip/pip values for orientation check
                    index_tip_y = lm[8].y
                    index_pip_y = lm[6].y
                    debug_lines.append(
                        f"index_tip_y: {index_tip_y:.3f}, index_pip_y: {index_pip_y:.3f}"
                    )
                    # optional: thumb orientation
                    thumb_tip_y = lm[4].y
                    thumb_ip_y = lm[3].y
                    debug_lines.append(
                        f"thumb_tip_y: {thumb_tip_y:.3f}, thumb_ip_y: {thumb_ip_y:.3f}"
                    )

                    # Gesture priority: palm -> fist -> index/thumb volume
                    if (
                        fingers == [1, 1, 1, 1, 1]
                        and (now - last_action_time) > cooldown
                    ):
                        gesture_text = "Play / Pause"
                        try:
                            pyautogui.press("playpause")
                        except Exception:
                            pyautogui.press("space")
                        last_action_time = now

                    elif (
                        fingers == [0, 0, 0, 0, 0]
                        and (now - last_action_time) > cooldown
                    ):
                        gesture_text = "Next Track"
                        try:
                            pyautogui.press("nexttrack")
                        except Exception:
                            pyautogui.hotkey("ctrl", "right")
                        last_action_time = now

                    # Index-only gestures: index extended, others closed
                    elif fingers == [0, 1, 0, 0, 0]:
                        # index pointing up -> tip.y < pip.y
                        if (
                            index_tip_y < index_pip_y
                            and (now - last_action_time) > cooldown
                        ):
                            gesture_text = "Volume Up (Index)"
                            ok = change_volume(+0.05)
                            debug_lines.append(f"change_volume + -> {ok}")
                            last_action_time = now
                        elif (
                            index_tip_y > index_pip_y
                            and (now - last_action_time) > cooldown
                        ):
                            gesture_text = "Volume Down (Index)"
                            ok = change_volume(-0.05)
                            debug_lines.append(f"change_volume - -> {ok}")
                            last_action_time = now

                    # Thumb-only gestures: thumb extended (others closed)
                    elif fingers == [1, 0, 0, 0, 0]:
                        if (
                            thumb_tip_y < thumb_ip_y
                            and (now - last_action_time) > cooldown
                        ):
                            gesture_text = "Volume Up (Thumb)"
                            ok = change_volume(+0.05)
                            debug_lines.append(f"thumb + -> {ok}")
                            last_action_time = now
                        elif (
                            thumb_tip_y > thumb_ip_y
                            and (now - last_action_time) > cooldown
                        ):
                            gesture_text = "Volume Down (Thumb)"
                            ok = change_volume(-0.05)
                            debug_lines.append(f"thumb - -> {ok}")
                            last_action_time = now

                    # draw index tip marker for clarity
                    idx_x = int(lm[8].x * w)
                    idx_y = int(lm[8].y * h)
                    cv2.circle(frame, (idx_x, idx_y), 8, (0, 255, 0), -1)

            else:
                debug_lines.append("no hand detected")

            # overlay detected gesture and debug
            if gesture_text:
                cv2.putText(
                    frame,
                    f"{gesture_text}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            debug_box.markdown("**Gesture debug**")
            debug_box.write("\n".join(debug_lines))
            # update current master volume shown
            vol_now = get_volume_scalar()
            debug_box.markdown(
                f"pycaw active: **{use_pycaw}**  \nMaster volume: **{vol_now:.2f}**"
                if vol_now is not None
                else f"pycaw active: **{use_pycaw}**  \nMaster volume: **N/A**"
            )

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
