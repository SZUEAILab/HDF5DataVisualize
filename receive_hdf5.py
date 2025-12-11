from flask import Flask, render_template, jsonify, request, send_file
import os
import h5py
import numpy as np
import cv2
import base64
import io
from PIL import Image
import json
import subprocess  # New import
import atexit  # New import
import signal  # New import
import uuid
from werkzeug.utils import secure_filename
from pyngrok import ngrok
import os
import hashlib
import tempfile

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'hdf5'}
# 用户上传的文件应该默认保存在哪个目录
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# 创建上传目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- RDT Process Management ---
rdt_process = None

upload_chunks = {}


@app.route('/api/upload_chunk', methods=['POST'])
def upload_chunk():
    """接收文件块"""
    try:
        if 'chunk' not in request.files:
            return jsonify({'error': '没有文件块'}), 400

        chunk_file = request.files['chunk']
        file_md5 = request.form.get('file_md5')
        filename = request.form.get('filename')
        chunk_index = int(request.form.get('chunk_index'))
        total_chunks = int(request.form.get('total_chunks'))

        if not all([file_md5, filename, chunk_index is not None, total_chunks]):
            return jsonify({'error': '缺少必要参数'}), 400

        # 创建临时目录存储块
        temp_dir = os.path.join(tempfile.gettempdir(), 'hdf5_uploads', file_md5)
        os.makedirs(temp_dir, exist_ok=True)

        # 保存块文件
        chunk_path = os.path.join(temp_dir, f'chunk_{chunk_index}')
        chunk_file.save(chunk_path)

        # 记录上传进度
        if file_md5 not in upload_chunks:
            upload_chunks[file_md5] = {
                'filename': filename,
                'total_chunks': total_chunks,
                'uploaded_chunks': set(),
                'temp_dir': temp_dir
            }

        upload_chunks[file_md5]['uploaded_chunks'].add(chunk_index)

        return jsonify({
            'success': True,
            'message': f'块 {chunk_index} 上传成功',
            'uploaded_chunks': len(upload_chunks[file_md5]['uploaded_chunks']),
            'total_chunks': total_chunks
        })

    except Exception as e:
        return jsonify({'error': f'上传块失败: {str(e)}'}), 500


@app.route('/api/merge_chunks', methods=['POST'])
def merge_chunks():
    """合并所有文件块"""
    try:
        data = request.get_json()
        file_md5 = data.get('file_md5')
        filename = data.get('filename')
        total_chunks = data.get('total_chunks')

        if file_md5 not in upload_chunks:
            return jsonify({'error': '文件不存在或已过期'}), 404

        chunk_info = upload_chunks[file_md5]

        # 检查是否所有块都已上传
        if len(chunk_info['uploaded_chunks']) != total_chunks:
            return jsonify({
                'error': f'块不完整 ({len(chunk_info["uploaded_chunks"])}/{total_chunks})'
            }), 400

        # 合并文件
        temp_dir = chunk_info['temp_dir']
        final_filename = f"{hashlib.md5(filename.encode()).hexdigest()}_{filename}"
        final_path = os.path.join('databases', 'uploaded_from_api', final_filename)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)

        with open(final_path, 'wb') as output_file:
            for i in range(total_chunks):
                chunk_path = os.path.join(temp_dir, f'chunk_{i}')
                with open(chunk_path, 'rb') as chunk_file:
                    output_file.write(chunk_file.read())
                # 删除临时块文件
                os.remove(chunk_path)

        # 删除临时目录
        os.rmdir(temp_dir)

        # 清理记录
        del upload_chunks[file_md5]

        return jsonify({
            'success': True,
            'message': '文件合并成功',
            'filename': final_filename,
            'file_path': final_path
        })

    except Exception as e:
        return jsonify({'error': f'合并文件失败: {str(e)}'}), 500
