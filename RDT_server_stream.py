# 文件路径: utils/Intergrated_interface_system/RDT_server_stream.py

#!/usr/bin/env python
# -- coding: UTF-8
"""
Web-streaming version of the RDT server.
This version correctly uses command-line arguments passed from the web backend.
"""
import os
import json
import argparse
import sys
import threading
import time
import yaml
from collections import deque
import numpy as np
import torch
from PIL import Image as PImage
import cv2
import socket
# This import works because the script is run with project root as cwd
from scripts.agilex_model_single import create_model
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import logging

# --- Video Streaming Globals ---
latest_wrist_frame = None
latest_third_frame = None
frame_lock = threading.Lock()
# --- End Video Streaming Globals ---

# Set up logging
logging.basicConfig(level=logging.INFO, format='[RDT_STREAM_SERVER] %(asctime)s - %(message)s')

# --- Functions (copied from RDT_server_hans.py) ---
# ... (所有函数如 make_policy, set_seed, etc. 在这里保持不变) ...
# 为了简洁，这里省略了函数体，请确保您文件中有这些函数的完整内容
CAMERA_NAMES = ['cam_wrist', 'cam_right_high', 'cam_left_high']
ACTION_BLOCK = 4
observation_window = None
lang_embeddings = None

def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config,
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )
    return model

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_config(args):
    return {'episode_len': args.max_publish_step, 'state_dim': 7, 'chunk_size': args.chunk_size, 'camera_names': CAMERA_NAMES}

def update_observation_window(args, config, state, img_controller):
    global observation_window
    def base64_to_image(base64_string):
        if not base64_string: return None
        return cv2.imdecode(np.frombuffer(base64.b64decode(base64_string), dtype=np.uint8), cv2.IMREAD_COLOR)
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        return cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    if observation_window is None:
        observation_window = deque(maxlen=2)
        observation_window.append({'qpos': None, 'images': {k: None for k in config["camera_names"]}})
    wrist_img = jpeg_mapping(base64_to_image(img_controller.get('wrist')))
    third_img = jpeg_mapping(base64_to_image(img_controller.get('third')))
    qpos = torch.from_numpy(np.array(state)).float().cuda()
    observation_window.append({'qpos': qpos, 'images': {config["camera_names"][0]: wrist_img, config["camera_names"][1]: None, config["camera_names"][2]: third_img}})

def inference_fn(args, config, policy):
    global observation_window, lang_embeddings
    time1 = time.time()
    image_arrs = [observation_window[-2]['images'][cam] for cam in config['camera_names']] + \
                 [observation_window[-1]['images'][cam] for cam in config['camera_names']]
    images = [PImage.fromarray(arr) if arr is not None else None for arr in image_arrs]
    proprio = observation_window[-1]['qpos'].unsqueeze(0)
    actions = policy.step(proprio=proprio, images=images, text_embeds=lang_embeddings).squeeze(0).cpu().numpy()
    logging.info(f"Model inference time: {time.time() - time1} s")
    return actions

def parse_http_request(data):
    try:
        header_end = data.find(b'\r\n\r\n')
        if header_end == -1: return None
        headers_part = data[:header_end].split(b'\r\n')
        body = data[header_end + 4:]
        content_length = 0
        for line in headers_part:
            if line.lower().startswith(b'content-length:'):
                content_length = int(line.split(b':')[1].strip())
                break
        if len(body) >= content_length:
            return body[:content_length].decode('utf-8')
    except Exception:
        return None
    return None

def visualize_images_for_stream(images_dict):
    global latest_wrist_frame, latest_third_frame, frame_lock
    def decode_and_encode_jpeg(base64_string):
        if not base64_string: return None
        try:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(base64_string), dtype=np.uint8), cv2.IMREAD_COLOR)
            return cv2.imencode('.jpg', img)[1].tobytes() if img is not None else None
        except Exception: return None
    with frame_lock:
        latest_wrist_frame = decode_and_encode_jpeg(images_dict.get('wrist'))
        latest_third_frame = decode_and_encode_jpeg(images_dict.get('third'))

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        stream_path = self.path.split('?')[0]
        frame_source = None
        if stream_path == '/wrist': frame_source = 'wrist'
        elif stream_path == '/third': frame_source = 'third'
        else:
            self.send_response(404); self.end_headers(); return
        
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--frame')
        self.end_headers()
        while True:
            try:
                with frame_lock:
                    frame = latest_wrist_frame if frame_source == 'wrist' else latest_third_frame
                if frame:
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                time.sleep(0.03)
            except Exception: break
            
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

def run_stream_server(host='0.0.0.0', port=8080):
    try:
        server = ThreadedHTTPServer((host, port), StreamingHandler)
        logging.info(f"MJPEG streaming server started at http://{host}:{port}")
        server.serve_forever()
    except Exception as e:
        logging.error(f"Could not start MJPEG streaming server: {e}")

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Arguments from original RDT_server_hans.py, reflecting inference_ym.sh usage
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", help='Path to the config file')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Name or path to the pretrained model')
    parser.add_argument('--lang_embeddings_path', type=str, required=True, help='Path to the pre-encoded language instruction embeddings')
    parser.add_argument('--ctrl_freq', action='store', type=int, default=25, help='The control frequency of the robot')
    
    # Other arguments that might be useful
    parser.add_argument('--max_publish_step', action='store', type=int, default=10000)
    parser.add_argument('--seed', action='store', type=int, default=None)
    parser.add_argument('--chunk_size', action='store', type=int, default=64)
    parser.add_argument('--HOST', type=str, default="192.168.0.115")
    parser.add_argument('--PORT', type=int, default=12345)
    
    args = parser.parse_args()

    # Start the MJPEG streaming server in a separate, non-blocking thread
    stream_server_thread = threading.Thread(target=run_stream_server)
    stream_server_thread.daemon = True
    stream_server_thread.start()

    # --- Main RDT Logic ---
    if args.seed is not None: set_seed(args.seed)

    config = get_config(args)
    policy = make_policy(args)
    
    # *** KEY CHANGE: Use the path from the argument, not a hardcoded one ***
    logging.info(f"Loading language embeddings from: {args.lang_embeddings_path}")
    lang_dict = torch.load(args.lang_embeddings_path)
    logging.info(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"]
    
    gripper = 0
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((args.HOST, args.PORT))
    server_socket.listen(1)
    logging.info(f"RDT command server listening on {args.HOST}:{args.PORT}...")

    index = 0
    batch = 3
    actions = None
    
    while True:
        try:
            conn, addr = server_socket.accept()
            with conn:
                data = conn.recv(1024 * 1024) # Increased buffer size for images
                if not data: continue
                
                json_str = parse_http_request(data)
                if not json_str: continue
                
                data_dict = json.loads(json_str)
                position = [float(x) for x in data_dict['position']]
                img = data_dict['images']
    
                visualize_images_for_stream(img)  

                if index % ACTION_BLOCK == 0:
                    state = np.concatenate((position[:3], position[3:], [gripper]))
                    update_observation_window(args, config, state, img)
                    actions = inference_fn(args, config, policy)

                if actions is None: continue
                
                action_chunk = actions[(index % ACTION_BLOCK)*batch:(index % ACTION_BLOCK + 1)*batch]
                action = np.sum(action_chunk, axis=0)
                
                action_dict = {"X": float(action[0]), "Y": float(action[1]), "Z": float(action[2]), "Rx": float(action[3]), "Ry": float(action[4]), "Rz": float(action[5]), "gripper": float(0)}
                
                response_body = json.dumps(action_dict)
                response_headers = (f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(response_body)}\r\n\r\n")
                conn.sendall((response_headers + response_body).encode('utf-8'))
                index += 1
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"Main server loop error: {e}", exc_info=True)
            time.sleep(1)
    
    server_socket.close()
    logging.info("Server stopped.")