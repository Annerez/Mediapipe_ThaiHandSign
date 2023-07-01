from tempfile import NamedTemporaryFile
import tempfile
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import streamlit as st
import mediapipe as mp
from utils import *
from datetime import datetime
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence , sentence , predictions , threshold = [] , [] , [] , 0.9
    frames = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)

            draw_styled_landmarks(image, results, mp_drawing, mp_holistic)

            keypoints = extract_keypoints(results)

            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 3:
                    sentence = sentence[-3:]

            cv2.rectangle(image, (0, 0), (640, 40), -1)
            fontpath = "./angsana.ttc"
            font = ImageFont.truetype(fontpath, 30)
            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)
            draw.text((30, 0), ' '.join(sentence), font=font, fill=(255, 255, 255, 255))
            image = np.array(img_pil)

            frames.append(image)
        cap.release()
    return frames

def save_uploaded_file(video_file):
    temp_file = NamedTemporaryFile(delete=False)
    temp_file.write(video_file.read())
    temp_file.close()
    return temp_file.name

def display_video(frames):
    video_placeholder = st.empty()

    for frame in frames:

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame_rgb)

        video_placeholder.image(image)

        time.sleep(0.025)

def download_predicted_video(frames):
    output_path = "predicted_video.mp4"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

    with open(output_path, "rb") as f:
        video_bytes = f.read()

    st.download_button("Download Predicted Video", data=video_bytes, file_name=f"processed_video_{timestamp}.mp4")

def main():
    st.title("Thai Sign Language Detection Model")
    st.markdown("""---""")
    st.subheader("เพื่อความแม่นยำของโมเดล กรุณาถ่ายวีดีโอแบบครึ่งตัว")
    st.subheader("ดูตัวอย่างได้ที่ https://www.th-sl.com/")
    global model
    global actions

    large = "Large Model (110 words)"
    medium = "Normal Model (30 words) (recommended)"
    test = "Test Model (4 words)"
    options = [test, medium, large]
    app_mode = st.sidebar.selectbox("Choose the app mode", options)

    st.subheader(app_mode)

    if app_mode == medium:
        actions = np.array(['สวัสดี', 'รอ', 'ขอบคุณ', 'กลับ', 'รัก', 'ตัด', 'หา', 'ขึ้น', 'ลง', 'ยก', 'พา', 'ช่วย', 'เชื่อ', 'คุย', 'ฟัง', 'สูง', 'ยืม', 'เอา', 'รู้จัก', 'บน', 'โกหก', 'กิน', 'กระโดด', 'รวม', 'วิ่ง', 'ไป', 'โง่', 'พักผ่อน', 'ได้ยิน'])
        model = load_model2("actions3.h5", actions)
    # elif app_mode == large:
    #     actions = np.array(['สวัสดี', 'ตก', 'รอ', 'กลับ', 'ขอบคุณ', 'ตัด', 'ลง', 'ขึ้น', 'เฝ้า','คุย', 'ช่วย', 'เชื่อ', 'ฟัง', 'มอง', 'พา', 'ชวน หลีกภัย', 'ทักษิณ ชินวัตร', 'ขนม', 'พิธา ลิ้มเจริญรัตน์', 'ศิริกัญญา ตันสกุล','แบก', 'อนุทิน ชาญวีรกูล', 'รังสิมันต์ โรม', 'พีระพันธุ์ สาลีรัฐวิภาค', 'คุณหญิงสุดารัตน์ เกยุราพันธุ์', 'พลเอกอนุพงษ์ เผ่าจินดา', 'สุวัจน์ ลิปตพัลลภ', 'กรณ์ จาติกวณิช', 'วราวุธ ศิลปะอาชา', 'พล.ต.อ.เสรีพิศุทธ์ เตมียเวสวราวุธ ศิลปะอาชา', 'ศักดิ์สยาม ชิดชอบ', 'ชาดา ไทยเศรษฐ์', 'สุชัชวีร์ สุวรรณสวัสดิ์', 'จุรินทร์ ลักษณวิศิษฏ์', 'ไตรรงค์ สุวรรณคีรี', 'พลเอกประยุทธ์ จันทร์โอชา', 'นฤมล ภิญโญสินวัฒน์', 'ธรรมนัส พรหมเผ่า', 'ชัยวุฒิ ธนาคมานุสรณ์', 'ไพบูลย์ นิติตะวัน', 'พลเอกประวิตร วงษ์สุวรรณ', 'พริษฐ์ วัชรสินธุ', 'ยิ่งลักษณ์ ชินวัตร', 'ณัฐวุฒิ ใสยเกื้อ', 'นพ.ชลน่าน ศรีแก้ว', 'เศรษฐา ทวีสิน', 'แพทองธาร ชินวัตร', 'อะไร', 'สมัคร', 'กระโดด', 'ยก', 'ชน', 'ผ่าน', 'แนะนำ', 'จำได้', 'ปลูก', 'รวม', 'ทำหาย', 'เจอ', 'หาย', 'พนักงานขาย', 'นักเขียนโปรแกรม', 'พ่อครัว', 'เจ้าหน้าที่ตำรวจ', 'เนื้อลูกแกะ', 'เนื้อหมู', 'คุณสบายดีไหม', 'ดูเหมือน', 'วาง', 'อยู่', 'ก', 'ข', 'ค', 'ฆ', 'ต', 'ถ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ'])
    #     model = load_model("action2.h5", actions)
    elif app_mode == test:
        actions = np.array(['สวัสดี', 'รอ', 'ขอบคุณ', 'กลับ'])
        model = load_model2("actions.h5", actions)

    video_file = st.file_uploader("Upload video file", type=["mp4"])
    if video_file is not None:
        video_path = save_uploaded_file(video_file)
        frames = process_video(video_path)

        download_predicted_video(frames)

        display_video(frames)


if __name__ == "__main__":
    main()
