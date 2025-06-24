import cv2
import numpy as np
from moviepy import VideoFileClip
import tempfile
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os

# Pydantic models for API documentation
class MinimapDetectionResult(BaseModel):
    is_minimap: bool
    circle_count: int
    color_score: float
    structure_score: float
    texture_complexity: float
    has_corners: bool

class SkillBarDetectionResult(BaseModel):
    is_skill_bar: bool
    round_icon_count: int
    bright_element_count: int

class ErrorResponse(BaseModel):
    error: str

# Enhanced LoL detection models
class EnhancedMinimapResult(BaseModel):
    is_minimap: bool
    circle_count: int
    small_elements: int
    color_score: float
    edge_density: float

class EnhancedSkillbarResult(BaseModel):
    is_skillbar: bool
    skill_icon_count: int
    round_element_count: int
    brightness_score: float
    background_darkness: float

class HealthbarResult(BaseModel):
    is_healthbar: bool
    green_ratio: float
    blue_ratio: float
    red_ratio: float

class ScoreboardResult(BaseModel):
    is_scoreboard: bool
    text_element_count: int
    brightness_ratio: float

class UIElementsResult(BaseModel):
    minimap_detected: EnhancedMinimapResult
    skillbar_detected: EnhancedSkillbarResult
    healthbar_detected: HealthbarResult
    scoreboard_detected: ScoreboardResult

class EnhancedFrameResult(BaseModel):
    time_point: float
    ui_elements: UIElementsResult
    legacy_minimap: MinimapDetectionResult
    legacy_skillbar: SkillBarDetectionResult
    is_lol_frame: bool

class FeatureScores(BaseModel):
    minimap_score: float
    skillbar_score: float
    healthbar_score: float
    scoreboard_score: float

class EnhancedVideoAnalysisResponse(BaseModel):
    is_lol_video: bool
    confidence: float
    feature_scores: FeatureScores
    frame_results: List[EnhancedFrameResult]

# LoL detection functions
def detect_minimap(frame):
    """실제 게임 화면에서 LoL 미니맵 감지"""
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

    """LoL 스킬바 감지"""
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


# # Minimap (우측 하단)
# minimap_rect = (int(w * 0.8), int(h * 0.8), w, h)

# # Skillbar (중앙 하단)
# skillbar_rect = (int(w * 0.3), int(h * 0.85), int(w * 0.7), h)

# # Scoreboard (상단 전체, 높이는 좁게)
# scoreboard_rect = (0, 0, w, int(h * 0.08))

# # 체력바 너비를 전체의 약 15%로 제한
# healthbar_width = int(w * 0.15)

# # 좌측 체력바: 좌측 전체 높이
# healthbar_left_rect = (0, 0, healthbar_width, h)

# # 우측 체력바: 우측 전체 높이
# healthbar_right_rect = (w - healthbar_width, 0, w, h)


# Enhanced LoL detection functions
def detect_lol_ui_elements(frame):
    """LoL UI 요소들을 감지 - 미니맵, 스킬바, 체력바, 점수판"""
    h, w, _ = frame.shape
    # 1. 미니맵 영역 (우측 하단)
    minimap_roi = frame[int(w*0.8):w, int(h*0.8):h]
    # 2. 스킬바 영역 (하단 중앙)
    skillbar_roi = frame[int(h*0.85):h, int(w*0.3):int(w*0.7)]
    # 3. 점수판 영역 (우상단)
    scoreboard_roi = frame[0:int(h*0.08), 0:w]

    # 체력바 너비를 전체의 약 15%로 제한
    healthbar_width = int(w * 0.15)
    # 좌측 체력바 ROI: 좌측 전체 높이
    healthbar_left_roi = frame[0:h, 0:healthbar_width]
    # 우측 체력바 ROI: 우측 전체 높이  
    healthbar_right_roi = frame[0:h, (w - healthbar_width):w]
    
    results = {
        "minimap_detected": detect_minimap_enhanced(minimap_roi),
        "skillbar_detected": detect_skillbar_enhanced(skillbar_roi),
        "healthbar_detected": {
            "left": detect_healthbar(healthbar_left_roi),
            "right": detect_healthbar(healthbar_right_roi)
        },
        "scoreboard_detected": detect_scoreboard(scoreboard_roi)
    }
    return results

def detect_minimap_enhanced(roi):
    """미니맵 감지"""
    if roi.size == 0:
        return {
            "is_minimap": False,
            "circle_count": 0,
            "small_elements": 0,
            "color_score": 0.0,
            "edge_density": 0.0
        }
    
    h, w, _ = roi.shape
    
    # LoL 미니맵 특징: 원형 구조, 파란색/녹색 지형, 작은 점들 (챔피언, 미니언)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 1. 원형 구조 감지
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=30, minRadius=10, maxRadius=min(h, w)//2
    )
    
    # 2. LoL 미니맵 색상 특징 (파란색 강, 녹색 정글)
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
    
    # 3. 작은 점들 감지 (챔피언, 미니언)
    small_circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=5,
        param1=30, param2=15, minRadius=2, maxRadius=8
    )
    
    # 4. 미니맵 테두리 감지 (보통 어두운 테두리)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    
    # 종합 판단
    has_main_circle = circles is not None and len(circles[0]) > 0
    has_color_features = (np.sum(blue_mask) + np.sum(green_mask)) / (h * w) > 0.1
    # has_small_elements = small_circles is not None and len(small_circles[0]) > 3
    # has_border = edge_density > 0.05
    
    return {
        "is_minimap": has_main_circle and has_color_features,
        "circle_count": len(circles[0]) if circles is not None else 0,
        "small_elements": len(small_circles[0]) if small_circles is not None else 0,
        "color_score": (np.sum(blue_mask) + np.sum(green_mask)) / (h * w),
        "edge_density": edge_density
    }

def detect_skillbar_enhanced(roi):
    """향상된 스킬바 감지 (LoL 특화)"""
    if roi.size == 0:
        return {
            "is_skillbar": False,
            "skill_icon_count": 0,
            "round_element_count": 0,
            "brightness_score": 0.0,
            "background_darkness": 0.0
        }
    
    h, w, _ = roi.shape
    
    # LoL 스킬바 특징: 4개의 원형 스킬 아이콘, D/F 스펠, 아이템 슬롯
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 1. 원형 스킬 아이콘 감지
    skill_circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
        param1=50, param2=25, minRadius=15, maxRadius=35
    )
    
    # 2. 밝은 색상 영역 (스킬 아이콘은 보통 밝음)
    bright_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))
    
    # 3. 컨투어 검출로 스킬 아이콘 모양 확인
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 원형에 가까운 컨투어 필터링
    round_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # 최소 크기
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)
            if circularity > 0.7:  # 원형에 가까운 것만
                round_contours.append(contour)
    
    # 4. 스킬바 배경 색상 (보통 어두운 회색)
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 100]))
    
    has_skill_icons = skill_circles is not None and len(skill_circles[0]) >= 3
    has_round_elements = len(round_contours) >= 3
    has_bright_elements = np.sum(bright_mask) / (h * w) > 0.1
    has_dark_background = np.sum(dark_mask) / (h * w) > 0.3
    
    return {
        "is_skillbar": has_skill_icons or (has_round_elements and has_bright_elements),
        "skill_icon_count": len(skill_circles[0]) if skill_circles is not None else 0,
        "round_element_count": len(round_contours),
        "brightness_score": np.sum(bright_mask) / (h * w),
        "background_darkness": np.sum(dark_mask) / (h * w)
    }

def detect_healthbar(roi):
    """체력바 감지"""
    if roi.size == 0:
        return {
            "is_healthbar": False,
            "green_ratio": 0.0,
            "blue_ratio": 0.0,
            "red_ratio": 0.0
        }
    
    h, w, _ = roi.shape
    
    # LoL 체력바 특징: 녹색(체력), 파란색(마나), 빨간색(적 체력)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 녹색 체력바
    green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
    
    # 파란색 마나바
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    
    # 빨간색 적 체력바
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = red_mask1 + red_mask2
    
    # 가로 방향 바 형태 감지
    green_ratio = np.sum(green_mask) / (h * w)
    blue_ratio = np.sum(blue_mask) / (h * w)
    red_ratio = np.sum(red_mask) / (h * w)
    
    has_healthbar = green_ratio > 0.05 or blue_ratio > 0.05 or red_ratio > 0.05
    
    return {
        "is_healthbar": has_healthbar,
        "green_ratio": green_ratio,
        "blue_ratio": blue_ratio,
        "red_ratio": red_ratio
    }

def detect_scoreboard(roi):
    """점수판 감지"""
    if roi.size == 0:
        return {
            "is_scoreboard": False,
            "text_element_count": 0,
            "brightness_ratio": 0.0
        }
    
    h, w, _ = roi.shape
    
    # LoL 점수판 특징: 작은 텍스트, 숫자, 팀 정보
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 텍스트 영역 감지 (작은 직사각형들)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 작은 직사각형 컨투어 필터링
    text_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 500:  # 텍스트 크기 범위
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            aspect_ratio = w_contour / (h_contour + 1e-5)
            if 0.5 < aspect_ratio < 3:  # 텍스트 비율
                text_contours.append(contour)
    
    # 밝은 텍스트 영역
    bright_pixels = np.sum(thresh > 0) / (h * w)
    
    return {
        "is_scoreboard": len(text_contours) > 5 and bright_pixels > 0.1,
        "text_element_count": len(text_contours),
        "brightness_ratio": bright_pixels
    }

def analyze_lol_features(video_path):
    """LoL 특화 특징들을 종합적으로 분석"""
    clip = None
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        
        # 여러 시점에서 프레임 분석
        analysis_points = [duration * 0.25, duration * 0.5, duration * 0.75]
        frame_results = []
        
        for time_point in analysis_points:
            frame = clip.get_frame(time_point)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # LoL UI 요소 분석
            ui_results = detect_lol_ui_elements(frame_bgr)
            
            # 기존 분석과 결합
            minimap_result = detect_minimap(frame_bgr)
            
            frame_result = {
                "time_point": time_point,
                "ui_elements": ui_results,
                "minimap": minimap_result,
                "is_lol_frame": (
                    ui_results["minimap_detected"]["is_minimap"] or
                    ui_results["skillbar_detected"]["is_skillbar"] or
                    ui_results["healthbar_detected"]["is_healthbar"] or
                    ui_results["scoreboard_detected"]["is_scoreboard"]
                )
            }
            frame_results.append(frame_result)
        
        # 종합 점수 계산
        positive_frames = sum(1 for r in frame_results if r["is_lol_frame"])
        confidence = positive_frames / len(frame_results)
        
        # 특징별 점수
        feature_scores = {
            "minimap_score": sum(1 for r in frame_results if r["ui_elements"]["minimap_detected"]["is_minimap"]) / len(frame_results),
            "skillbar_score": sum(1 for r in frame_results if r["ui_elements"]["skillbar_detected"]["is_skillbar"]) / len(frame_results),
            "healthbar_score": sum(1 for r in frame_results if r["ui_elements"]["healthbar_detected"]["is_healthbar"]) / len(frame_results),
            "scoreboard_score": sum(1 for r in frame_results if r["ui_elements"]["scoreboard_detected"]["is_scoreboard"]) / len(frame_results)
        }
        
        return {
            "is_lol_video": confidence > 0.3,  # 더 관대한 임계값
            "confidence": confidence,
            "feature_scores": feature_scores,
            "frame_results": frame_results
        }
        
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception as e:
                print(f"Warning: Could not close video file in analyze_lol_features: {e}")

app = FastAPI(
    title="LoL Video Analyzer API",
    description="""
# League of Legends Video Analysis API

This API analyzes uploaded video files to determine if they contain League of Legends gameplay.

## Features

- **Minimap Detection**: Identifies LoL minimap elements in video frames
- **Skill Bar Detection**: Detects LoL skill bar interface elements  
- **Multi-frame Analysis**: Analyzes multiple frames for higher accuracy
- **Confidence Scoring**: Provides confidence levels for analysis results

## Supported Video Formats

- WebM (recommended)
- MP4
- AVI
- MOV
- MKV

## How it works

1. Upload a video file using the `/analyze` endpoint
2. The API extracts multiple frames from the video
3. Each frame is analyzed for LoL-specific UI elements
4. Results include detection confidence and detailed analysis

## API Endpoints

- `POST /analyze` - Upload and analyze video files
- `GET /health` - Health check endpoint
- `GET /` - API information

## Usage

Visit `/docs` for interactive API documentation with Swagger UI.
    """,
    version="1.0.0",
    contact={
        "name": "LoL Video Analyzer",
        "email": "support@lolanalyzer.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

@app.post(
    "/analyze",
    response_model=EnhancedVideoAnalysisResponse,
    responses={
        200: {
            "description": "Video analysis completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "is_lol_video": True,
                        "confidence": 0.85,
                        "feature_scores": {
                            "minimap_score": 0.8,
                            "skillbar_score": 0.9,
                            "healthbar_score": 0.7,
                            "scoreboard_score": 0.6
                        },
                        "frame_results": [
                            {
                                "time_point": 15.5,
                                "ui_elements": {
                                    "minimap_detected": {
                                        "is_minimap": True,
                                        "circle_count": 3,
                                        "small_elements": 8,
                                        "color_score": 0.15,
                                        "edge_density": 0.08
                                    },
                                    "skillbar_detected": {
                                        "is_skillbar": True,
                                        "skill_icon_count": 4,
                                        "round_element_count": 6,
                                        "brightness_score": 0.25,
                                        "background_darkness": 0.45
                                    },
                                    "healthbar_detected": {
                                        "is_healthbar": True,
                                        "green_ratio": 0.12,
                                        "blue_ratio": 0.08,
                                        "red_ratio": 0.05
                                    },
                                    "scoreboard_detected": {
                                        "is_scoreboard": True,
                                        "text_element_count": 12,
                                        "brightness_ratio": 0.18
                                    }
                                },
                                "legacy_minimap": {
                                    "is_minimap": True,
                                    "circle_count": 5,
                                    "color_score": 0.15,
                                    "structure_score": 0.25,
                                    "texture_complexity": 45.2,
                                    "has_corners": True
                                },
                                "legacy_skillbar": {
                                    "is_skill_bar": True,
                                    "round_icon_count": 4,
                                    "bright_element_count": 4
                                },
                                "is_lol_frame": True
                            }
                        ]
                    }
                }
            }
        },
        400: {
            "description": "Invalid file format or corrupted video",
            "model": ErrorResponse
        },
        413: {
            "description": "File too large",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error during analysis",
            "model": ErrorResponse
        }
    },
    summary="Analyze video for League of Legends gameplay",
    description="""
# Video Analysis Endpoint

Upload and analyze a video file to determine if it contains League of Legends gameplay.

## Analysis Features

- **Enhanced Minimap Detection**: Identifies LoL minimap elements using advanced computer vision
- **Enhanced Skill Bar Detection**: Detects LoL skill bar interface elements with improved accuracy
- **Health Bar Detection**: Identifies LoL health and mana bars
- **Scoreboard Detection**: Detects LoL scoreboard and team information
- **Multi-frame Analysis**: Analyzes 3 frames from different time points (25%, 50%, 75%)
- **Feature Scoring**: Provides detailed confidence scores for each UI element

## Parameters

- `file`: Video file to analyze (WebM, MP4, AVI, MOV, MKV supported)

## Returns

- `is_lol_video`: Boolean indicating if the video contains LoL gameplay
- `confidence`: Overall confidence score (0.0 to 1.0) for the analysis
- `feature_scores`: Individual confidence scores for each UI element:
  - `minimap_score`: Confidence in minimap detection
  - `skillbar_score`: Confidence in skill bar detection
  - `healthbar_score`: Confidence in health bar detection
  - `scoreboard_score`: Confidence in scoreboard detection
- `frame_results`: Detailed analysis results for each analyzed frame including:
  - `time_point`: Time in video when frame was analyzed
  - `ui_elements`: Enhanced UI element detection results
  - `legacy_minimap`: Original minimap detection results
  - `legacy_skillbar`: Original skill bar detection results
  - `is_lol_frame`: Whether this specific frame contains LoL elements

## Enhanced Detection Features

### Minimap Detection
- Circular structure detection
- Blue/green terrain color analysis
- Small element detection (champions, minions)
- Edge density analysis

### Skill Bar Detection
- Circular skill icon detection
- Bright element analysis
- Background darkness detection
- Contour-based shape analysis

### Health Bar Detection
- Green health bar detection
- Blue mana bar detection
- Red enemy health bar detection
- Color ratio analysis

### Scoreboard Detection
- Text element detection
- Brightness ratio analysis
- Small rectangular contour filtering
    """,
    tags=["Video Analysis"]
)
async def analyze(file: UploadFile = File(..., description="Video file to analyze for LoL gameplay")):
    """
    Analyze uploaded video for League of Legends gameplay detection.
    
    This endpoint performs comprehensive analysis of video content to identify
    League of Legends gameplay elements using both legacy and enhanced detection methods.
    
    The enhanced analysis includes:
    - Advanced minimap detection with circular structure analysis
    - Improved skill bar detection with contour analysis
    - Health bar detection for player and enemy health/mana
    - Scoreboard detection for team information
    - Multi-frame analysis at 25%, 50%, and 75% of video duration
    
    Args:
        file: The video file to analyze
        
    Returns:
        EnhancedVideoAnalysisResponse: Detailed analysis results with confidence scores
        
    Raises:
        HTTPException: If file format is unsupported or analysis fails
    """
    # Validate file type
    allowed_extensions = {'.webm', '.mp4', '.avi', '.mov', '.mkv'}
    
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )
    
    file_extension = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (limit to 100MB)
    max_size = 100 * 1024 * 1024  # 100MB
    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 100MB."
        )
    
    try:
        # Create temporary file with proper cleanup
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            # Write file content in chunks to avoid memory issues
            shutil.copyfileobj(file.file, temp_file)
            temp_file.close()  # Explicitly close the file handle
            tmp_path = temp_file.name

            result = analyze_lol_features(tmp_path)
            
        finally:
            # Clean up temporary file
            if temp_file and hasattr(temp_file, 'name'):
                try:
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                except Exception as cleanup_error:
                    print(f"Warning: Could not delete temporary file {temp_file.name}: {cleanup_error}")
        
        return result

    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file' in locals() and temp_file and hasattr(temp_file, 'name'):
            try:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
            except Exception as cleanup_error:
                print(f"Warning: Could not delete temporary file {temp_file.name}: {cleanup_error}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get(
    "/health",
    summary="Health check endpoint",
    description="""
# Health Check

Simple health check to verify the API is running and responsive.

## Returns

Returns a JSON object with the current status of the API service.

## Use Cases

- **Monitoring**: Check if the API is operational
- **Load Balancers**: Health check for load balancer configuration
- **DevOps**: Automated monitoring and alerting
    """,
    tags=["System"]
)
async def health_check():
    """
    Health check endpoint to verify API status.
    
    Returns:
        dict: API status information
    """
    return {
        "status": "healthy",
        "service": "LoL Video Analyzer API",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)