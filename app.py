import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- إعدادات الصفحة ---
st.set_page_config(page_title="نظام ترجمة لغة الإشارة", layout="wide")

# إعدادات الاتصال للكاميرا (ضرورية للنشر على الويب)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- محرك التعرف على الإشارات (نفس المنطق الخاص بك) ---
class GestureEngine:
    def __init__(self):
        self.rules = []
        self._register()

    def _features(self, landmarks, handed):
        pts = [np.array([p.x, p.y, p.z], dtype=np.float32) for p in landmarks]
        wrist, idx_mcp, pnk_mcp = pts[0], pts[5], pts[17]
        v1, v2 = idx_mcp - wrist, pnk_mcp - wrist
        palm_normal = np.cross(v1, v2)
        
        facing = "امام" if palm_normal[2] < -0.12 else "خلف" if palm_normal[2] > 0.12 else "محايد"

        def is_open(f): return pts[f[3]][1] < pts[f[1]][1]
        
        fingers = {
            "ابهام": pts[4][0] < pts[3][0] if handed == "Left" else pts[4][0] > pts[3][0],
            "سبابة": is_open([5,6,7,8]), "وسطى": is_open([9,10,11,12]),
            "بنصر": is_open([13,14,15,16]), "خنصر": is_open([17,18,19,20])
        }
        return {"open": fingers, "facing": facing}

    def _register(self):
        self.rules.append(("سلام", lambda f: all(f["open"].values())))
        self.rules.append(("توقف", lambda f: f["facing"]=="امام" and f["open"]["سبابة"]))
        self.rules.append(("نصر", lambda f: f["open"]["سبابة"] and f["open"]["وسطى"] and not f["open"]["بنصر"]))
        self.rules.append(("أنا", lambda f: f["open"]["سبابة"] and not f["open"]["وسطى"]))

    def classify(self, landmarks, handed):
        f = self._features(landmarks, handed)
        for n, fn in self.rules:
            try:
                if fn(f): return n
            except: continue
        return "جاري التحليل..."

# --- معالج الفيديو (Video Processor) ---
class VideoProcessor:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.engine = GestureEngine()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            handed = res.multi_handedness[0].classification[0].label
            label = self.engine.classify(hand.landmark, handed)
            
            # رسم الهيكل العظمي
            mp.solutions.drawing_utils.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)
            # كتابة النتيجة على الصورة
            cv2.putText(img, label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        return frame.from_ndarray(img, format="bgr24")

# --- واجهة المستخدم (Streamlit UI) ---

# شريط جانبي للأسماء
st.sidebar.title("🎓 تفاصيل المشروع")
st.sidebar.markdown("""
### إعداد الطالبات:
* **شهد صادق حمزة**
* **بنين عبد الله عبد الزهرة**
* **فاطمة كريم حميد شبيب**

### إشراف:
* **الست زهراء كاظم فرهود**
""")

st.title("✨ نظام ترجمة لغة الإشارة (AI)")
st.write("مرحباً بكم في نظامنا الذكي للتعرف على لغة الإشارة العربية.")

# منطقة الكاميرا
webrtc_streamer(
    key="sign-lang",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},
)

st.info("💡 اضغط على 'Start' لتشغيل الكاميرا وابدأ بالإشارة بيدك.")
