# --- ระบบปฏิบัติการและพื้นฐาน (Sysstem & Utilities) ---
import os          # จัดการไฟล์/โฟลเดอร์ (เช่น สร้างที่เก็บรูปภาพ OK/NG)
import ctypes      # เรียกใช้ไฟล์ Library (.so/.dll) ระดับต่ำของระบบ
import time        # ใช้หน่วงเวลา และจับเวลาการทำงาน (Inference time)
import json        # จัดการไฟล์ config หรือบันทึกการตั้งค่าต่างๆ
import socket      # การเชื่อมต่อเครือข่าย TCP/IP (ส่งข้อมูลข้ามเครื่อง)
from datetime import datetime # จัดการวันที่และเวลา (ใช้ทำ log และชื่อไฟล์)

# --- การประมวลผลภาพและ AI (Vision & AI Core) ---
import cv2           # OpenCV: วาดกรอบ Bounding Box, ใส่ตัวหนังสือ และจัดการภาพ
import numpy as np # คำนวณตัวเลขเชิงลึกของภาพ (Array) เพื่อความเร็ว
from ultralytics import YOLO # โหลดโมเดล YOLOv8 เพื่อใช้ตรวจจับชิ้นงาน
from PIL import Image        # จัดการไฟล์รูปภาพเพื่อนำไปแสดงผลผ่าน GUI

# --- การควบคุมฮาร์ตแวร์ (Hardware Control) ---
from MvImport.MvCameraControl_class import * # ควบคุมกล้อง Hikrobot (MvImport)
import threading   # ทำงานแบบขนาน (เช่น แยก thread ดึงภาพออกจาก Thread ประมวลผล)

# --- ส่วนแสดงผลหน้าจอ (User Interface) ---
import customtkinter as ctk # สร้างหน้าจอ GUI

# =============================================
# SETTING & CONFIG FILE
# ====ชชชช=====================================
# 🚩 ตั้งชื่อโปรแกรมหลัก (GUI DIsplay)
APP_TITLE = "AI INSPECTION: R_RED ULTIMATE V5"
HEADR_TEXT = "R-RED MONITORING SYSTEM"

# 🚩 ตั้งค่าที่อยู่ Path
MODEL_PATH = '/home/its/my_yolo_project/runs/detect/R_RED_ULTIMATE_5060_110526/weights/best.pt'
CONFIG_FILE = "threshold_config.json" # ไฟล์เก็บความจำ

# 🚩 กรอบแสดงภาพ
ROI_Y1, ROI_Y2 = 380, 920 # default 380,920
ROI_X1, ROI_X2 = 300, 1600 # default 300,1600

# 🚩 ตั้งค่า PLC
PLC_IP, PLC_PORT = '192.168.10.12', 9600

# 🚩 ตั้งค่าเก็บ Log ภาพประวัติ
SAVE_DIR = "R_RED_inspection_history_V5"

# 🚩 ตั้งค่ากล้อง
EXPOSURE_TIME = 4000.0
GAIN_VALUE = 10.0
TRIGGER_MODE = "OFF"

# สร้าง folder SAVE_DIR ถ้ายังไม่มี
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
# ==============================================

class VisionApp(ctk.CTk):
    def __init__(self):
        # สร้าง หน้าต่างโปรแกรม
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1700x1000")
        self.configure(fg_color= "#c7c9cd")

        # เตรียมทรัพยากรระบบ (ที่อยู่,การ์ดจอ ,ขนาดกรอบภาพสำหรับแสดง)
        print(f"🔃 Loading Model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH).to('cuda:0')
        self.digit_gallery = {str(i): {'img': np.zeros((100, 80, 3 ), np.uint8), 'color': "#64748b"} for i in range(10)}
        self.running = True
        self.sharpen_kernel = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])

        # --- [ส่วนที่เพิ่ม] ตัวแปรสำหรับระบบ Counter และ Cycle Time ---
        self.count_total = 0
        self.count_ok = 0
        self.count_ng = 0
        self.current_piece_counted = False
        self.last_d_class = "None"

        self.start_time_all = time.perf_counter()
        self.loop_start_time = 0.0
        self.val_last_cycle = 0.0
        self.val_max_cycle = 0.0
        self.val_avg_cycle = 0.0
        self.is_first_piece_of_loop = True

        # --- ระบบโหลดค่าความจำ ---
        self.thresh_vars = {}
        saved_data = self.load_config()

        # --- ตั้งค่าพื้นฐาน ---
        # --- ตั้งค่า NG นอกกลุ่มตัวเลข ---
        default_keys = ["R_POLE_NG", "R_GEAR_NG", "R_CLEAR", "R_DOT_NG", "DEFAULT"]
        for key in default_keys:
            val = saved_data.get(key, 0.90 if "NG" in key else 0.75)
            self.thresh_vars[key] = ctk.DoubleVar(value=val)

        # --- ตั้งค่ากลุ่มตัวเลข ---
        for i in range(10):
            key = f"R_{i}_NG"
            val = saved_data.get(key, 0.90)
            self.thresh_vars[key] = ctk.DoubleVar(value=val)

        # --- แสดงข้อมูล ui ที่ window ---
        self.setup_ui()
        threading.Thread(target=self.process_camera, daemon=True).start()
        threading.Thread(target=self.auto_sync_plc_trigger, daemon=True).start()

    # --- ดึงการตั้งค่าของเดิมมาใช้ ---
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except: return {}
        return {}

    def save_config(self, *args):
        data = {key: var.get() for key, var in self.thresh_vars.items()}
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saveing config: {e}")

    def setup_ui(self):
        self.left_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.left_panel.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        self.lbl_title = ctk.CTkLabel(self.left_panel, text="R_RED MONITOR (MEMORY MODE)", font=("Arial", 32, "bold"), text_color= "#38bdf8")
        self.lbl_title.pack(pady=(0,10))

        self.video_label = ctk.CTkLabel(self.left_panel, text="", fg_color= "#FFFFFF", corner_radius=10)
        self.video_label.pack(pady=10, fill="both", expand=True)

        self.status_bar = ctk.CTkFrame(self.left_panel, height=80, fg_color= "#3f5578", corner_radius=15)
        self.status_bar.pack(fill="x", pady=10)

        self.lbl_detect = ctk.CTkLabel(self.status_bar, text="WAITING...", font=("Arial", 24, "bold"), text_color= "#94a3b8")
        self.lbl_detect.pack(side="left", padx=40)

        self.lbl_msg = ctk.CTkLabel(self.status_bar, text="SYSTEM READY", font=("Arial", 24, "bold"), text_color= "#22c55e")
        self.lbl_msg.pack(side="right", padx=40)

        self.gallery_frame = ctk.CTkFrame(self.left_panel, height=150, fg_color= "#D8DAE3", corner_radius=15)
        self.gallery_frame.pack(fill="x", pady=10)
        self.gallery_slots = []
        for i in range(10):
            slot = ctk.CTkLabel(self.gallery_frame, text=f"{i}", font=("Arial", 12), compound="top", text_color= "#64748b")
            slot.pack(side="left", expand=True, padx=5, pady=10)
            self.gallery_slots.append(slot)

        self.right_sidebar = ctk.CTkScrollableFrame(self, width=380, fg_color= "#5b5c5e" ,label_text= "THRESHOLD SETTING")
        self.right_sidebar.pack(side="right", fill="y", padx=(0,20), pady=20)

        for key, var in sorted(self.thresh_vars.items()):
            f = ctk.CTkFrame(self.right_sidebar, fg_color= "#CECFD0")
            f.pack(fill="x", padx=5, pady=3)
            ctk.CTkLabel(f, text=f"{key}", font=("Arial", 11, "bold")).pack(side="left", padx=10)
            v_lbl = ctk.CTkLabel(f, text=f"{var.get():.2f}", text_color= "#e32828")
            v_lbl.pack(side="right", padx=10)
            s = ctk.CTkSlider(self.right_sidebar, from_=0.1, to=0.99, variable=var,
                              command=lambda v, l=v_lbl: [l.configure(text=f"{float(v):.2f}"), self.save_config()])
            s.pack(fill="x", padx=15, pady=(0,10))

# --- Counter bar (ปรับปรุงเพื่อให้รองรับ Cycle Time) ---
        self.bottom_bar = ctk.CTkFrame(self.left_panel, height=120, fg_color="#1e293b", corner_radius=15)
        self.bottom_bar.pack(fill="x", side="bottom", pady=(10, 0))

        # 1. TARGET (Entry)
        self.target_frame = ctk.CTkFrame(self.bottom_bar, fg_color="transparent")
        self.target_frame.pack(side="left", padx=20, expand=True)
        ctk.CTkLabel(self.target_frame, text="TARGET", font=("Arial", 12, "bold"), text_color="#94a3b8").pack()
        self.entry_target = ctk.CTkEntry(self.target_frame, width=80, font=("Arial", 20, "bold"), justify="center")
        self.entry_target.insert(0, "1000")
        self.entry_target.pack()
        self.entry_target.bind("<Return>", self.update_target_event)

        # 2. TOTAL
        self.total_frame = ctk.CTkFrame(self.bottom_bar, fg_color="transparent")
        self.total_frame.pack(side="left", padx=20, expand=True)
        ctk.CTkLabel(self.total_frame, text="TOTAL", font=("Arial", 12, "bold"), text_color="#ffffff").pack()
        self.lbl_total_cnt = ctk.CTkLabel(self.total_frame, text="0", font=("Arial", 28, "bold"), text_color="#ffffff")
        self.lbl_total_cnt.pack()

        # 3. OK
        self.ok_frame = ctk.CTkFrame(self.bottom_bar, fg_color="transparent")
        self.ok_frame.pack(side="left", padx=20, expand=True)
        ctk.CTkLabel(self.ok_frame, text="OK", font=("Arial", 12, "bold"), text_color="#22c55e").pack()
        self.lbl_ok_cnt = ctk.CTkLabel(self.ok_frame, text="0", font=("Arial", 28, "bold"), text_color="#22c55e")
        self.lbl_ok_cnt.pack()

        # 4. NG
        self.ng_frame = ctk.CTkFrame(self.bottom_bar, fg_color="transparent")
        self.ng_frame.pack(side="left", padx=20, expand=True)
        ctk.CTkLabel(self.ng_frame, text="NG", font=("Arial", 12, "bold"), text_color="#ef4444").pack()
        self.lbl_ng_cnt = ctk.CTkLabel(self.ng_frame, text="0", font=("Arial", 28, "bold"), text_color="#ef4444")
        self.lbl_ng_cnt.pack()

        # --- ส่วนแสดงเวลา (Cycle Time) ---
        self.avg_time_frame = ctk.CTkFrame(self.bottom_bar, fg_color="transparent")
        self.avg_time_frame.pack(side="left", padx=20, expand=True)
        ctk.CTkLabel(self.avg_time_frame, text="AVG CT", font=("Arial", 12, "bold"), text_color="#38bdf8").pack()
        self.lbl_avg_time = ctk.CTkLabel(self.avg_time_frame, text="0.00", font=("Arial", 24, "bold"), text_color="#38bdf8")
        self.lbl_avg_time.pack()

        self.last_time_frame = ctk.CTkFrame(self.bottom_bar, fg_color="transparent")
        self.last_time_frame.pack(side="left", padx=20, expand=True)
        ctk.CTkLabel(self.last_time_frame, text="LAST CT", font=("Arial", 12, "bold"), text_color="#fbbf24").pack()
        self.lbl_last_time = ctk.CTkLabel(self.last_time_frame, text="0.00", font=("Arial", 24, "bold"), text_color="#fbbf24")
        self.lbl_last_time.pack()

        self.max_time_frame = ctk.CTkFrame(self.bottom_bar, fg_color="transparent")
        self.max_time_frame.pack(side="left", padx=20, expand=True)
        ctk.CTkLabel(self.max_time_frame, text="MAX CT", font=("Arial", 12, "bold"), text_color="#f87171").pack()
        self.lbl_max_time = ctk.CTkLabel(self.max_time_frame, text="0.00", font=("Arial", 24, "bold"), text_color="#f87171")
        self.lbl_max_time.pack()

    def update_target_event(self, even=None):
        try:
            target_val = int(self.entry_target.get())
            self.lbl_msg.configure(text=f"เป้าหมาย: {target_val}", text_color="#38bdf8")
            self.focus()
        except ValueError:
            self.lbl_msg.configure(text="ERROR: กรอกเฉพาะตัวเลข", text_color="#ef4444")

    def get_threshold_live(self, d_class):
        target_var = self.thresh_vars.get(d_class, self.thresh_vars["DEFAULT"])
        return target_var.get()

    def process_camera(self):
        cam = None # 🚩 เพิ่มเพื่อป้องกัน cam not defined
        deviceList = MV_CC_DEVICE_INFO_LIST()
        MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
        if deviceList.nDeviceNum == 0: return

        cam = MvCamera()
        stDeviceUint = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        cam.MV_CC_CreateHandle(stDeviceUint)
        cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        cam.MV_CC_SetFloatValue("ExposureTime", EXPOSURE_TIME)
        cam.MV_CC_SetFloatValue("Gain", GAIN_VALUE)

        if TRIGGER_MODE == "OFF":
            cam.MV_CC_SetEnumValue("TriggerMode", 0)
        else:
            cam.MV_CC_SetEnumValue("TriggerMode", 1)
            cam.MV_CC_SetEnumValue("TriggerSource", 7)

        cam.MV_CC_StartGrabbing()
        stOutFrame = MV_FRAME_OUT()

        while self.running:
            ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1500)
            if ret == 0:
                pData = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                ctypes.memmove(pData, stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
                img_raw = np.frombuffer(pData, dtype=np.uint8).reshape(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth)

                img_color = cv2.cvtColor(img_raw, cv2.COLOR_BayerBG2BGR)
                img_sharpen = cv2.filter2D(img_color, -1, self.sharpen_kernel)
                img_roi = img_sharpen[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()
                img_display = img_roi.copy()
                h_roi, w_roi = img_roi.shape[:2]

                results = self.model.predict(img_roi, imgsz=1280, conf=0.2, verbose=False, device=0)
                d_class, is_pass, msg = "None", True, "PASS"

                central_boxes = []
                if len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        b = box.xyxy[0].cpu().numpy().astype(int)
                        center_x = (b[0] + b[2]) // 2
                        if (w_roi * 0.3 < center_x < w_roi * 0.7):
                            central_boxes.append(box)
                        else:
                            cv2.rectangle(img_display, (b[0], b[1]), (b[2], b[3]), (150, 150, 150), 1)

                    for box in central_boxes:
                        cls_name = self.model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        b = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), min(w_roi, b[2]), min(h_roi, b[3])
                        crop_box = img_roi[y1:y2, x1:x2]
                        has_bad_color = False

                        if crop_box.size > 0:
                            hsv_box = cv2.cvtColor(crop_box, cv2.COLOR_BGR2HSV)
                            black_mask = cv2.inRange(hsv_box, np.array([0,0,0]), np.array([180, 255, 5]))
                            if cv2.countNonZero(black_mask) > 100: has_bad_color = True

                        limit = self.get_threshold_live(cls_name)

                        # --- 🚩 ปรับจังหวะ Logic REJECT ใหม่ (แยกกลุ่ม NG และ OK ออกจากกัน) ---
                        is_ng_class = "_NG" in cls_name or cls_name == "R_CLEAR"
                        
                        if is_ng_class or has_bad_color:
                            if has_bad_color or conf >= limit:
                                is_pass = False
                                d_class = f"{cls_name}_CLR_ERR" if has_bad_color else cls_name
                                msg = f"REJECT: {d_class}"
                                cv2.rectangle(img_display, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 4)
                                cv2.putText(img_display, d_class, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                break

                        # Logic PASS (เขียว) - ย้าย GEAR/POLE ปกติมาเช็คที่นี่
                        elif ("_OK" in cls_name or "GEAR" in cls_name or "POLE" in cls_name) and is_pass:
                            d_class = cls_name
                            cv2.rectangle(img_display, (b[0], b[1]), (b[2], b[3]), (34, 197, 94), 2)
                            cv2.putText(img_display, cls_name, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 197, 94), 1)
                            for n in range(10):
                                if f"R_{n}_OK" == cls_name and crop_box.size > 0:
                                    rotated_img = cv2.rotate(crop_box, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                    self.digit_gallery[str(n)]['img'] = rotated_img.copy()
                                    self.digit_gallery[str(n)]['color'] = "#22c55e"

                # --- 🚩 Logic จับเวลาและนับจำนวน (แทรกตามต้นฉบับ) ---
                if len(central_boxes) > 0:
                    # จังหวะเจอชิ้นงานตัวแรกของรอบ
                    if self.is_first_piece_of_loop:
                        now = time.perf_counter()
                        if self.loop_start_time > 0:
                            self.val_last_cycle = now - self.loop_start_time
                            if self.val_last_cycle > self.val_max_cycle: self.val_max_cycle = self.val_last_cycle
                        self.loop_start_time = now
                        self.is_first_piece_of_loop = False

                    # นับจำนวนกรณี NG
                    if not is_pass and not self.current_piece_counted:
                        self.count_total += 1
                        self.count_ng += 1
                        self.current_piece_counted = True
                        for n in range(10):
                            if str(n) in d_class: self.digit_gallery[str(n)]['color'] = "#ef4444"
                        threading.Thread(target=self.trigger_reject, args=(d_class, img_roi.copy())).start()
                    self.last_d_class = d_class
                else:
                    # จังหวะพ้นชิ้นงาน (นับ OK)
                    if not self.current_piece_counted and self.last_d_class != "None":
                        self.count_total += 1
                        self.count_ok += 1
                        self.current_piece_counted = True
                        self.last_d_class = "None"
                    self.current_piece_counted = False
                    self.is_first_piece_of_loop = True

                self.after(0, self.update_ui, img_display, d_class, is_pass, msg)
                cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                time.sleep(0.01)

        # 🚩 ปิดกล้องเมื่อเลิกใช้งาน (เพิ่ม if cam เพื่อกันพัง)
        if cam is not None:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()

    def update_ui(self, img_display, d_class, is_pass, msg):
        # อัปเดตสถิติตัวเลข
        self.lbl_total_cnt.configure(text=str(self.count_total))
        self.lbl_ok_cnt.configure(text=str(self.count_ok))
        self.lbl_ng_cnt.configure(text=str(self.count_ng))
        if self.count_total > 0:
            self.val_avg_cycle = (time.perf_counter() - self.start_time_all) / self.count_total
            self.lbl_avg_time.configure(text=f"{self.val_avg_cycle:.2f}")
            self.lbl_last_time.configure(text=f"{self.val_last_cycle:.2f}")
            self.lbl_max_time.configure(text=f"{self.val_max_cycle:.2f}")

        h, w = img_display.shape[:2]
        display_w = 1100
        display_h = int(h * (display_w / w))
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.video_label.configure(image=ctk.CTkImage(img_pil, size=(display_w, display_h)))

        self.lbl_detect.configure(text=f"DETECT: {d_class}", text_color="#f8fafc")
        is_not_digit = not any(str(i) in d_class for i in range(10))
        if not is_pass and is_not_digit and d_class != "None":
            self.lbl_msg.configure(text=f"⚠️ EJECT: {d_class}", text_color="#ef4444")
        else:
            self.lbl_msg.configure(text=msg, text_color="#22c55e" if is_pass else "#ef4444")

        for i in range(10):
            item = self.digit_gallery[str(i)]
            if np.sum(item['img']) > 0:
                g_img = Image.fromarray(cv2.cvtColor(item['img'], cv2.COLOR_BGR2RGB))
                self.gallery_slots[i].configure(image=ctk.CTkImage(g_img, size=(60, 90)),
                                               fg_color=item['color'], text_color="white")

    def trigger_reject(self, d_class, img_save):
            try:
                now = datetime.now()
                date_str = now.strftime('%-d-%-m-%y_%H%M%S')
                filename = f"{date_str}_{d_class}_ng.png"
                cv2.imwrite(f"{SAVE_DIR}/{filename}", img_save, [cv2.IMWRITE_PNG_COMPRESSION, 1])

                header = [0x80, 0x00, 0x02, 0x00, 12, 0x00, 0x00, 251, 0x00, 0x01]
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.settimeout(0.2)
                    s.sendto(bytearray(header + [0x01, 0x02, 0xB1, 0x00, 0x1E, 0x00, 0x00, 0x01, 0x00, 0x01]), (PLC_IP, PLC_PORT))
                    time.sleep(0.12)
                    s.sendto(bytearray(header + [0x01, 0x02, 0xB1, 0x00, 0x1E, 0x00, 0x00, 0x01, 0x00, 0x00]), (PLC_IP, PLC_PORT))
            except: pass

    def auto_sync_plc_trigger(self):
        while self.running:
            try:
                header = [0x80, 0x00, 0x02, 0x00, 12, 0x00, 0x00, 251, 0x00, 0x01]
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.settimeout(0.1)
                    s.sendto(bytearray(header + [0x01, 0x01, 0x31, 0x00, 0x0C, 0x00, 0x00, 0x01]), (PLC_IP, PLC_PORT))
                    res, _ = s.recvfrom(1024)
                    if len(res) >= 15 and res[14] == 0x01:
                        for addr in [0x78, 0xDC]:
                            s.sendto(bytearray(header + [0x01, 0x02, 0x82, 0x00, addr, 0x00, 0x00, 0x02, 0,0,0,0]), (PLC_IP, PLC_PORT))
            except: pass
            time.sleep(0.05)
            


if __name__ == "__main__":
    app = VisionApp()
    app.mainloop()
