import cv2
import numpy as np
from moviepy import VideoFileClip
import tempfile
import shutil
from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse

def extract_frame(video_path, time_sec=1):
    """비디오에서 특정 시간의 프레임을 추출합니다."""
    clip = VideoFileClip(video_path)
    frame = clip.get_frame(time_sec)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def detect_minimap(frame):
    """실제 게임 화면에서 LoL 미니맵을 감지합니다."""
    h, w, _ = frame.shape

    # 미니맵이 있을 것으로 예상되는 영역 (화면 오른쪽 하단)
    roi_size = min(h, w) // 5  # 화면 크기에 따라 동적으로 조정
    roi = frame[h-roi_size:h, w-roi_size:w]

    # 특징 추출을 위한 전처리
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # 1. 색상 특징 분석 - HSV 색상 공간에서 파란색 및 녹색 영역 감지
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 파란색 강 감지 (85-130)
    lower_blue = np.array([85, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)

    # 녹색/황색 아군 요소 감지 (35-85)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)

    # 2. 테두리 및 구조 분석
    # 적응형 임계값 처리로 미니맵 테두리 강화
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 모폴로지 연산으로 노이즈 제거 및 특징 강화
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

    # 3. 코너 검출 (미니맵은 사각형 구조를 가짐)
    corners = cv2.goodFeaturesToTrack(morph, 25, 0.01, 10)
    has_corners = corners is not None and len(corners) >= 4

    # 4. 원형 객체 감지 (챔피언, 와드, 미니언 등)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=15,
        param1=50,
        param2=25,  # 더 낮은 값으로 설정하여 감도 향상
        minRadius=3,
        maxRadius=20
    )

    has_circles = circles is not None
    circle_count = 0 if not has_circles else len(circles[0])

    # 5. 미니맵 구조 분석 - 명암비, 질감 특성
    # 표준 편차로 질감의 복잡성 측정
    texture_complexity = np.std(gray_roi)

    # 6. 종합 점수 계산
    color_score = (np.sum(blue_mask > 0) + np.sum(green_mask > 0)) / (roi_size * roi_size)
    structure_score = np.sum(morph > 0) / (roi_size * roi_size)

    # 최종 판단 (각 특징에 가중치 부여)
    is_minimap = (
            color_score > 0.01 and  # 파란색/녹색 요소 존재
            structure_score > 0.1 and  # 구조적 특징 존재
            texture_complexity > 20 and  # 일정 수준 이상의 질감 복잡성
            has_corners and  # 코너 존재
            circle_count >= 1  # 최소 하나 이상의 원형 객체 존재
    )

    result = {
        "is_minimap": bool(is_minimap),
        "circle_count": int(circle_count),
        "color_score": float(color_score),
        "structure_score": float(structure_score),
        "texture_complexity": float(texture_complexity),
        "has_corners": bool(has_corners)
    }

    return result

def detect_skill_bar(frame):
    """LoL 스킬바를 감지합니다."""
    h, w, _ = frame.shape
    roi = frame[h-150:h-50, int(w*0.25):int(w*0.75)]

    # 색상 변환
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 이진화 처리
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # 윤곽선 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 원형에 가까운 객체 추출 (스킬 아이콘)
    roundish = [c for c in contours if
                cv2.contourArea(c) > 50 and
                cv2.arcLength(c, True)**2 / (4 * np.pi * cv2.contourArea(c) + 1e-5) < 1.5]

    # 밝은 색상 객체 감지 (스킬 아이콘은 보통 밝게 표시됨)
    bright_mask = cv2.inRange(hsv_roi, np.array([0, 0, 150]), np.array([180, 30, 255]))
    bright_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 스킬바 감지 결과
    has_skill_icons = len(roundish) >= 3
    has_bright_elements = len([c for c in bright_contours if cv2.contourArea(c) > 30]) >= 3

    result = {
        "is_skill_bar": has_skill_icons or has_bright_elements,
        "round_icon_count": len(roundish),
        "bright_element_count": len([c for c in bright_contours if cv2.contourArea(c) > 30])
    }

    return result

def is_lol_video(video_path):
    """비디오가 LoL 게임 영상인지 판별합니다."""
    frame = extract_frame(video_path)
    minimap_result = detect_minimap(frame)
    skill_bar_result = detect_skill_bar(frame)

    return {
        "is_lol_video": minimap_result["is_minimap"] or skill_bar_result["is_skill_bar"],
        "minimap_detected": minimap_result,
        "skill_bar_detected": skill_bar_result
    }

def get_multiple_frames(video_path, num_frames=3):
    """비디오에서 여러 프레임을 추출하여 보다 정확한 분석을 수행합니다."""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frames = []

    # 영상의 다양한 시점에서 프레임 추출
    for i in range(num_frames):
        time_point = duration * (i + 1) / (num_frames + 1)
        frame = clip.get_frame(time_point)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return frames

def analyze_video(video_path):
    """여러 프레임을 분석하여 보다 정확한 결과를 제공합니다."""
    frames = get_multiple_frames(video_path, num_frames=3)
    results = []

    for frame in frames:
        minimap_result = detect_minimap(frame)
        skill_bar_result = detect_skill_bar(frame)

        frame_result = {
            "is_lol_frame": minimap_result["is_minimap"] or skill_bar_result["is_skill_bar"],
            "minimap": minimap_result,
            "skill_bar": skill_bar_result
        }
        results.append(frame_result)

    # 종합 결과 계산
    positive_frames = sum(1 for r in results if r["is_lol_frame"])
    confidence = positive_frames / len(results)

    return {
        "is_lol_video": confidence > 0.5,  # 50% 이상의 프레임이 LoL로 감지되면 LoL 영상으로 판단
        "confidence": confidence,
        "frame_results": results
    }

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """업로드된 비디오를 분석하여 LoL 게임 영상인지 판별합니다."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result = analyze_video(tmp_path)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)