from ultralytics import YOLOWorld

# 1. โหลดโมเดล
print("⏳ กำลังโหลดโมเดล...")
model = YOLOWorld('yolov8s-worldv2.pt')

# 2. รายการสิ่งของ (ตัดคนออก เพิ่มของใช้)
classes = [
    # เครื่องเขียน
    "pen", "marker", "pencil", "notebook", "book", "scissors", "stapler",
    # อุปกรณ์คอม/โต๊ะทำงาน
    "computer monitor", "laptop", "mouse", "keyboard", "headphones", "glasses",
    # ของกิน/เครื่องดื่ม
    "can", "soda can", "cup", "water bottle", "coffee mug", "plate", "bowl", "spoon", "fork",
    # ของใช้ส่วนตัว
    "keys", "wallet", "smartphone", "bag", "watch", "remote control",
    # ของในบ้าน
    "fan", "chair", "lamp", "trash can"
]
model.set_classes(classes)

# 3. Export เป็น ONNX (ขนาด 320x320 เพื่อความลื่นบนมือถือ)
print("⏳ กำลังสร้างสมอง (ONNX)...")
model.export(format='onnx', opset=12, imgsz=320)

print("✅ เสร็จแล้ว! คุณจะได้ไฟล์ 'yolov8s-worldv2.onnx'")