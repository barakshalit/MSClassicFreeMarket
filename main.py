import cv2
import easyocr
import re
import requests
import json
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
from dotenv import load_dotenv # NEW

# --- CONFIGURATION ---
load_dotenv() # NEW
API_KEY = os.environ.get("API_KEY")
SCRIPT_URL = os.environ.get("GOOGLE_SHEET")
PROCESS_EVERY_X_SECONDS = 1 

def get_video_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Select MapleStory OBS Recording",
        filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.flv *.mov"), ("All Files", "*.*")]
    )
    return file_path

def process_video():
    video_path = get_video_file()
    
    if not video_path:
        print("No file selected. Exiting.")
        return

    print(f"Loading EasyOCR model for: {video_path}...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview", 1280, 720)
    print("\n--- PREVIEW MODE ---")
    print("Press 'F' to Fast-Forward 2 seconds.")
    print("Press 'SPACE' when a store is open to pause and select areas.")
    
    target_frame = None
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Reached the end of the video without selecting.")
            return
            
        cv2.imshow("Preview", frame)
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord(' '):  
            target_frame = frame
            break
        elif key == ord('f') or key == ord('F'):  
            fps = cap.get(cv2.CAP_PROP_FPS)
            current = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current + (fps * 2))

    cv2.destroyWindow("Preview")

    print("\nSTEP 1: Draw a rectangle over the '3% fee' warning box at the bottom.")
    cv2.namedWindow("1. Select 3% Fee Box", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1. Select 3% Fee Box", 1280, 720)
    fee_roi = cv2.selectROI("1. Select 3% Fee Box", target_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("1. Select 3% Fee Box")

    print("STEP 2: Draw a rectangle over the LEFT ITEM COLUMN.")
    cv2.namedWindow("2. Select Store Item Box", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("2. Select Store Item Box", 1280, 720)
    item_roi = cv2.selectROI("2. Select Store Item Box", target_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("2. Select Store Item Box")
    
    fx, fy, fw, fh = int(fee_roi[0]), int(fee_roi[1]), int(fee_roi[2]), int(fee_roi[3])
    ix, iy, iw, ih = int(item_roi[0]), int(item_roi[1]), int(item_roi[2]), int(item_roi[3])

    if fw == 0 or iw == 0:
        print("Selection cancelled. Exiting.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_frame = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_jump = int(fps * PROCESS_EVERY_X_SECONDS)
    
    unique_items_batch = set()
    print("\nScanning video from the beginning...")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret: break 
            
        fee_crop = frame[fy:fy+fh, fx:fx+fw]
        fee_gray = cv2.cvtColor(fee_crop, cv2.COLOR_BGR2GRAY)
        fee_text_data = reader.readtext(fee_gray, detail=0)
        fee_text = " ".join(fee_text_data).lower()
        
        if "3" not in fee_text and "fee" not in fee_text:
            current_frame += frame_jump
            continue
            
        item_crop = frame[iy:iy+ih, ix:ix+iw]
        item_gray = cv2.cvtColor(item_crop, cv2.COLOR_BGR2GRAY)
        
        # 1. Upscale the image 2x so the letters are larger
        large_gray = cv2.resize(item_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 2. Apply CLAHE to locally darken faint gray text without blowing out the black text
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(large_gray)
        
        # 3. Read the locally enhanced image
        ocr_results = reader.readtext(enhanced_gray)
        
        # Pass the original unmodified item_gray for accurate SOLD color detection
        items = parse_maple_text(ocr_results, item_gray)
        for item in items:
            unique_items_batch.add((item['name'], item['price'], item['status']))
            
        current_frame += frame_jump

    cap.release()
    
    if unique_items_batch:
        payload = [{"name": i[0], "price": i[1], "status": i[2]} for i in unique_items_batch]
        send_to_sheets(payload)
    else:
        print("No items detected.")

def parse_maple_text(ocr_results, gray_img):
    items = []
    
    ocr_results.sort(key=lambda x: x[0][0][1])
    
    lines_info = []
    for bbox, text, prob in ocr_results:
        text = text.strip()
        if not text: continue
        
        # Divide bounding box coordinates by 2 because we upscaled the OCR image 2x
        y_min = int(min(p[1] for p in bbox) / 2)
        y_max = int(max(p[1] for p in bbox) / 2)
        x_min = int(min(p[0] for p in bbox) / 2)
        x_max = int(max(p[0] for p in bbox) / 2)
        
        y_min, y_max = max(0, y_min), min(gray_img.shape[0], y_max)
        x_min, x_max = max(0, x_min), min(gray_img.shape[1], x_max)
        
        region = gray_img[y_min:y_max, x_min:x_max]
        if region.size == 0: continue
        
        text_darkness = np.percentile(region, 10)
        is_sold = text_darkness > 140
        
        lines_info.append({"text": text, "is_sold": is_sold})
        
    for i in range(1, len(lines_info)):
        line = lines_info[i]["text"]
        price_match = re.search(r'(?:for\s*)?([\d\s,]+)\s*meso', line, re.IGNORECASE)
        
        if price_match:
            raw_price_str = re.sub(r'[\s,]', '', price_match.group(1))
            if not raw_price_str.isdigit():
                continue
                
            price_num = int(raw_price_str)
            raw_name = lines_info[i-1]["text"]
            
            clean_name = re.sub(r'[^a-zA-Z0-9\s\'-]', ' ', raw_name)
            clean_name = re.sub(r'^(\d+|[a-zA-Z]{1,2})\s+', '', clean_name).strip()
            
            is_sold_final = lines_info[i]["is_sold"] or lines_info[i-1]["is_sold"]
            status = "SOLD" if is_sold_final else "LISTED"
            
            if len(clean_name) >= 3 and price_num > 0:
                items.append({"name": clean_name, "price": price_num, "status": status})
                print(f"Found: {clean_name} -> {price_num} ({status})")
                
    return items

def send_to_sheets(items):
    print(f"\nSending {len(items)} unique records to Database...")
    
    # Bundle the token and the data together
    payload = {
        "token": os.environ.get("MAPLE_API_KEY"),
        "items": items
    }
    
    try:
        response = requests.post(
            SCRIPT_URL, 
            data=json.dumps(payload), # Send the bundled payload
            headers={"Content-Type": "text/plain;charset=utf-8"}
        )
        print("Server Response:", response.text)
    except Exception as e:
        print("Failed to send data:", e)

if __name__ == "__main__":
    process_video()