import cv2
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from backend.main import SubtitleRemover  # 你的 SubtitleRemover 类
from threading import Thread

app = Flask(__name__)

# 用一个字典来存储进度
progress_data = {
    'progress': 0,
    'status': 'Waiting...',
}

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route('/')
def ds():
    # return render_template('index.html')
    return render_template('ds.html')

def extract_sub_area_from_mask(mask_path):
    # 使用 OpenCV 加载蒙版
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 计算蒙版的外接矩形区域 (ymin, ymax, xmin, xmax)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min, y_min, w, h = cv2.boundingRect(contours[0])  # 获取第一个轮廓的外接矩形
    x_max = x_min + w
    y_max = y_min + h

    return (x_min, x_max, y_min, y_max)

def extract_first_frame(video_path):
    # 提取视频的第一帧
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        first_frame_path = os.path.join(OUTPUT_FOLDER, 'first_frame.jpg')
        cv2.imwrite(first_frame_path, frame)  # 保存为jpg
        cap.release()
        return first_frame_path
    cap.release()
    return None

# 获取视频分辨率
def get_video_resolution(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

# 获取图片分辨率
def get_image_resolution(image_path):
    # 读取图片
    img = cv2.imread(image_path)
    # 获取图片的宽度和高度
    height, width, _ = img.shape
    return width, height

# 获取蒙版相对坐标
def get_mask_coordinates(mask_path):
    # 读取蒙版图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 将蒙版图像二值化，确保是黑白图像
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 找到蒙版图像中的白色区域轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 获取最大的轮廓（假设只有一个区域需要擦除）
    max_contour = max(contours, key=cv2.contourArea)

    # 获取包围最大轮廓的矩形框
    x, y, w, h = cv2.boundingRect(max_contour)

    # 转换为相对坐标 (ymin, ymax, xmin, xmax)
    ymin = y / mask.shape[0]
    ymax = (y + h) / mask.shape[0]
    xmin = x / mask.shape[1]
    xmax = (x + w) / mask.shape[1]

    return ymin, ymax, xmin, xmax



# def scale_sub_area(sub_area, video_width, video_height, mask_width, mask_height):
#     """
#     根据蒙版尺寸将框选区域的坐标进行等比例放大
#     :param sub_area: 原始框选区域 (ymin, ymax, xmin, xmax)，值为蒙版坐标系的相对值
#     :param video_width: 视频宽度
#     :param video_height: 视频高度
#     :param mask_width: 蒙版宽度
#     :param mask_height: 蒙版高度
#     :return: 等比例放大的框选区域 (ymin, ymax, xmin, xmax)，值为视频坐标系的绝对值
#     """
#     ymin, ymax, xmin, xmax = sub_area
#
#
#     # 将蒙版区域的相对坐标转换为视频分辨率中的绝对坐标
#     xmin = round(xmin * video_width)
#     xmax = int(xmax * video_width)
#     ymin = int(ymin * video_height)
#     ymax = int(ymax * video_height)
#
#
#     return (xmin, xmax, ymin, ymax)
def scale_coordinates(vid_w, vid_h, mask_w, mask_h, rel_coords):
    # 计算缩放比例
    scale = min(vid_w / mask_w, vid_h / mask_h)

    # 计算缩放后蒙版尺寸
    scaled_w = mask_w * scale
    scaled_h = mask_h * scale

    # 计算居中偏移量
    offset_x = (vid_w - scaled_w) / 2
    offset_y = (vid_h - scaled_h) / 2

    # 解包相对坐标
    ymin_rel, ymax_rel, xmin_rel, xmax_rel  = rel_coords

    # 计算实际坐标
    xmin = round(offset_x + xmin_rel * scaled_w)
    xmax = round(offset_x + xmax_rel * scaled_w)
    ymin = round(offset_y + ymin_rel * scaled_h)
    ymax = round(offset_y + ymax_rel * scaled_h)

    return (ymin, ymax, xmin, xmax)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_path, mask_path
    if 'video_file' not in request.files:
        return jsonify({'error': 'No video file part'}), 400
    video_file = request.files['video_file']
    if video_file.filename == '':
        return jsonify({'error': 'No selected video file'}), 400

    # 保存视频文件
    video_filename = secure_filename(video_file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    video_file.save(video_path)

    sub_area = None
    if 'mask_file' in request.files:
        mask_file = request.files['mask_file']
        if mask_file.filename == '':
            return jsonify({'error': 'No selected mask file'}), 400
        # 保存蒙版文件
        mask_filename = secure_filename(mask_file.filename)
        mask_path = os.path.join(UPLOAD_FOLDER, mask_filename)
        mask_file.save(mask_path)



        # 获取字幕擦除区域
        video_width, video_height = get_video_resolution(video_path)
        print(f"视频分辨率:{video_width}×{video_height}")
        mask_width, mask_height = get_image_resolution(mask_path)
        print(f"图片分辨率:{mask_width}×{mask_height}")
        sub_area_init = get_mask_coordinates(mask_path)
        print(f"蒙版相对坐标：{sub_area_init}")
        sub_area_old = extract_sub_area_from_mask(mask_path)
        print(f"原始蒙版区域：{sub_area_old}")
        sub_area = scale_coordinates(video_width, video_height, mask_width, mask_height, sub_area_init)
        # sub_area = (282, 773, 1451, 1560)
        print(f"放缩后蒙版区域：{sub_area}")

    # 提取视频的第一帧
    first_frame_path = extract_first_frame(video_path)
    # print(first_frame_path)

    # 启动处理任务
    def process_video():
        global progress_data
        progress_data['progress'] = 0
        progress_data['status'] = 'Processing started...'

        # 调用 SubtitleRemover
        sr = SubtitleRemover(video_path, sub_area=sub_area)
        sr.run_with_progress_callback(update_progress)

        progress_data['progress'] = 100
        progress_data['status'] = 'Processing completed!'

    def update_progress(progress):
        global progress_data
        progress_data['progress'] = progress

    # 在后台线程中处理视频
    Thread(target=process_video, daemon=True).start()

    return jsonify({
        'status': 'Started processing...',
        'progress': 0,
        'first_frame_path': first_frame_path
    })

@app.route('/get_progress', methods=['GET'])
def get_progress():
    return jsonify(progress_data)

@app.route('/first_frame/<filename>')
def send_first_frame(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
