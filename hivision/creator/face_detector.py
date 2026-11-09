#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2024/9/5 19:32
@File: face_detector.py
@IDE: pycharm
@Description:
    人脸检测器
"""
try:
    from mtcnnruntime import MTCNN
except ImportError:
    raise ImportError(
        "Please install mtcnn-runtime by running `pip install mtcnn-runtime`"
    )
from .context import Context
from hivision.error import FaceError, APIError
from hivision.utils import resize_image_to_kb_base64
from hivision.creator.retinaface import retinaface_detect_faces
import logging
import requests
import cv2
import os
import numpy as np


mtcnn = None
base_dir = os.path.dirname(os.path.abspath(__file__))
RETINAFCE_SESS = None


def _mtcnn_detect(image, **kwargs):
    """Normalise mtcnn-runtime's no-face result across package versions."""
    try:
        return mtcnn.detect(image, **kwargs)
    except (ValueError, cv2.error) as exc:
        # mtcnn-runtime currently raises ValueError when its internal result list
        # is empty.  Treat that as the normal, user-actionable "no face" case —
        # but log the real exception so an install/version fault is not silently
        # misreported to the user as "no face in the photo" (see logs/studio.log).
        logging.getLogger("dz_photoid_studio").warning(
            "mtcnn detect raised %s: %s -- treated as no-face",
            type(exc).__name__, exc)
        return None, None


def _face_count(faces):
    return 0 if faces is None else len(faces)


def detect_face_mtcnn(ctx: Context, scale: int = 2):
    """
    基于MTCNN模型的人脸检测处理器，只进行人脸数量的检测
    :param ctx: 上下文，此时已获取到原始图和抠图结果，但是我们只需要原始图
    :param scale: 最大边长缩放比例，原图:缩放图 = 1:scale
    :raise FaceError: 人脸检测错误，多个人脸或者没有人脸
    """
    global mtcnn
    if mtcnn is None:
        mtcnn = MTCNN()
    if ctx.origin_image is None or getattr(ctx.origin_image, "ndim", 0) < 2:
        raise FaceError("Input image is missing or invalid", 0)
    height, width = ctx.origin_image.shape[:2]
    if height < 2 or width < 2:
        raise FaceError("Input image is too small for face detection", 0)
    scale = max(1, int(scale))
    image = cv2.resize(
        ctx.origin_image,
        (max(1, width // scale), max(1, height // scale)),
        interpolation=cv2.INTER_AREA,
    )
    # landmarks 是 5 个关键点，分别是左眼、右眼、鼻子、左嘴角、右嘴角，
    faces, landmarks = _mtcnn_detect(image, thresholds=[0.8, 0.8, 0.8])

    if _face_count(faces) != 1:
        # 保险措施，如果检测到多个人脸或者没有人脸，用原图再检测一次
        # The retry must use the SAME thresholds as the first pass; falling back
        # to the library defaults (0.6/0.7/0.8) silently changes the contract.
        faces, landmarks = _mtcnn_detect(ctx.origin_image,
                                         thresholds=[0.8, 0.8, 0.8])
    else:
        # Restore only coordinate values.  The old code multiplied every field,
        # including the confidence score, and assumed an exact 2x reduction.
        faces = np.array(faces, copy=True)
        if faces.ndim != 2 or faces.shape[1] < 5:
            # Unexpected detector output shape (unpinned mtcnn-runtime): fail as
            # a clean FaceError instead of crashing on faces[0, 0] indexing.
            raise FaceError("Expected 1 face, but got 0", 0)
        scale_x = width / image.shape[1]
        scale_y = height / image.shape[0]
        faces[0, 0] *= scale_x
        faces[0, 2] *= scale_x
        faces[0, 1] *= scale_y
        faces[0, 3] *= scale_y
        if landmarks is not None:
            landmarks = np.array(landmarks, copy=True)
            if landmarks.ndim >= 2 and landmarks.shape[0] and landmarks.shape[1] >= 10:
                landmarks[0, :5] *= scale_x
                landmarks[0, 5:10] *= scale_y
    face_num = _face_count(faces)
    if face_num != 1:
        raise FaceError("Expected 1 face, but got {}".format(face_num), face_num)

    # 计算人脸坐标
    left = faces[0][0]
    top = faces[0][1]
    width = faces[0][2] - left + 1
    height = faces[0][3] - top + 1
    ctx.face["rectangle"] = (left, top, width, height)

    # 根据landmarks计算人脸偏转角度，以眼睛为标准，计算的人脸偏转角度，用于人脸矫正
    # 示例landmarks [106.37181  150.77415  127.21012  108.369156 144.61522  105.24723 107.45625  133.62355  151.24269  153.34407 ]
    roll_angle = 0.0
    if landmarks is not None:
        landmarks = np.asarray(landmarks)
        if landmarks.ndim >= 2 and landmarks.shape[0] and landmarks.shape[1] >= 10:
            landmark = landmarks[0]
            left_eye = np.array([landmark[0], landmark[5]])
            right_eye = np.array([landmark[1], landmark[6]])
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            roll_angle = np.degrees(np.arctan2(dy, dx))

    ctx.face["roll_angle"] = roll_angle


def detect_face_face_plusplus(ctx: Context):
    """
    基于Face++ API接口的人脸检测处理器，只进行人脸数量的检测
    :param ctx: 上下文，此时已获取到原始图和抠图结果，但是我们只需要原始图
    :param scale: 最大边长缩放比例，原图:缩放图 = 1:scale
    :raise FaceError: 人脸检测错误，多个人脸或者没有人脸
    """
    url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
    api_key = os.getenv("FACE_PLUS_API_KEY")
    api_secret = os.getenv("FACE_PLUS_API_SECRET")

    if not api_key or not api_secret:
        raise APIError("Face++ API credentials are not configured.", 401)

    print("调用了face++")

    image = ctx.origin_image
    # 将图片转为 base64, 且不大于2MB（Face++ API接口限制）
    image_base64 = resize_image_to_kb_base64(image, 2000, mode="max")

    files = {
        "api_key": (None, api_key),
        "api_secret": (None, api_secret),
        "image_base64": (None, image_base64),
        "return_landmark": (None, "1"),
        "return_attributes": (None, "headpose"),
    }

    # 发送 POST 请求
    try:
        response = requests.post(url, files=files, timeout=30)
    except requests.RequestException as exc:
        raise APIError(f"Face++ request failed: {exc}", 0) from exc

    # 获取响应状态码
    status_code = response.status_code
    try:
        response_json = response.json()
    except ValueError:
        response_json = {}

    if status_code == 200:
        faces = response_json.get("faces") or []
        try:
            face_num = int(response_json.get("face_num", len(faces)))
        except (TypeError, ValueError):
            raise APIError("Face++ returned an invalid face count.", status_code)
        if face_num == 1 and faces:
            try:
                face_rectangle = faces[0]["face_rectangle"]

                # 获取人脸关键点
                # landmarks = response_json["faces"][0]["landmark"]
                # print("face++ landmarks", landmarks)

                # headpose 是一个字典，包含俯仰角（pitch）、偏航角（yaw）和滚转角（roll）
                # headpose示例 {'pitch_angle': 6.997899, 'roll_angle': 1.8011835, 'yaw_angle': 5.043002}
                headpose = faces[0]["attributes"]["headpose"]
                # 以眼睛为标准，计算的人脸偏转角度，用于人脸矫正
                roll_angle = headpose["roll_angle"] / 2

                ctx.face["rectangle"] = (
                    face_rectangle["left"],
                    face_rectangle["top"],
                    face_rectangle["width"],
                    face_rectangle["height"],
                )
                ctx.face["roll_angle"] = roll_angle
            except (IndexError, KeyError, TypeError) as exc:
                raise APIError("Face++ returned an incomplete success response.", status_code) from exc
        else:
            raise FaceError("Expected 1 face, but got {}".format(face_num), face_num)

    elif status_code == 401:
        raise APIError(
            f"Face++ Status code {status_code} Authentication error: API key and secret do not match.",
            status_code,
        )

    elif status_code == 403:
        reason = response_json.get("error_message", "Unknown authorization error.")
        raise APIError(
            f"Authorization error: {reason}",
            status_code,
        )

    elif status_code == 400:
        error_message = response_json.get("error_message", "Bad request.")
        raise APIError(
            f"Bad request error: {error_message}",
            status_code,
        )

    elif status_code == 413:
        raise APIError(
            f"Face++ Status code {status_code} Request entity too large: The image exceeds the 2MB limit.",
            status_code,
        )
    else:
        reason = response_json.get("error_message", "Unknown Face++ error.")
        raise APIError(f"Face++ Status code {status_code}: {reason}", status_code)


def detect_face_retinaface(ctx: Context):
    """
    基于RetinaFace模型的人脸检测处理器，只进行人脸数量的检测
    :param ctx: 上下文，此时已获取到原始图和抠图结果，但是我们只需要原始图
    :raise FaceError: 人脸检测错误，多个人脸或者没有人脸
    """
    from time import time

    global RETINAFCE_SESS

    if RETINAFCE_SESS is None:
        # 计算用时
        tic = time()
        faces_dets, sess = retinaface_detect_faces(
            ctx.origin_image,
            os.path.join(base_dir, "retinaface/weights/retinaface-resnet50.onnx"),
            sess=None,
        )
        RETINAFCE_SESS = sess
    else:
        tic = time()
        faces_dets, _ = retinaface_detect_faces(
            ctx.origin_image,
            os.path.join(base_dir, "retinaface/weights/retinaface-resnet50.onnx"),
            sess=RETINAFCE_SESS,
        )

    faces_dets = [] if faces_dets is None else faces_dets
    faces_num = len(faces_dets)
    faces_landmarks = []
    for face_det in faces_dets:
        faces_landmarks.append(face_det[5:])

    if faces_num != 1:
        raise FaceError("Expected 1 face, but got {}".format(faces_num), faces_num)
    face_det = faces_dets[0]
    ctx.face["rectangle"] = (
        face_det[0],
        face_det[1],
        face_det[2] - face_det[0] + 1,
        face_det[3] - face_det[1] + 1,
    )

    # 计算roll_angle
    face_landmarks = faces_landmarks[0]
    # print("face_landmarks", face_landmarks)
    left_eye = np.array([face_landmarks[0], face_landmarks[1]])
    right_eye = np.array([face_landmarks[2], face_landmarks[3]])
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll_angle = np.degrees(np.arctan2(dy, dx))
    ctx.face["roll_angle"] = roll_angle

    # 如果RUN_MODE不是野兽模式，则释放模型
    if os.getenv("RUN_MODE") != "beast":
        RETINAFCE_SESS = None
