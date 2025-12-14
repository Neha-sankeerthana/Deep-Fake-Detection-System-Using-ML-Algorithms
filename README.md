# Deep-Fake-Detection-System-Using-ML-Algorithms
Flask-based Deepfake Detection System with secure login, image &amp; video upload, and AI-simulated analysis. Features modern UI, confidence-based results, frame-level video stats, and a demo live detection module. Built for learning, demos, and UI-focused ML projects.
import os
import cv2
import numpy as np
import json
import random
from flask import Flask, render_template_string, request, redirect, Response, url_for, session, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import time

# Initialize Flask app
app = Flask(_name_)
app.secret_key = 'deepfake_detection_secret_key_2024'

# Configure upload settings
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "avif", "jpg", "jpeg", "mp4", "avi", "mov"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def mock_predict_image(image_path):
    """Mock prediction for image analysis"""
    # Simulate processing time
    time.sleep(random.uniform(1, 3))
    
    # Generate realistic mock results
    is_fake = random.choice([True, False])
    confidence = random.uniform(75, 98)
    
    return {
        "result": "Fake" if is_fake else "Real",
        "confidence": round(confidence, 2),
        "analysis_time": datetime.now().strftime("%H:%M:%S"),
        "file_type": "image"
    }

def mock_predict_video(video_path):
    """Mock prediction for video analysis"""
    # Simulate longer processing time for video
    time.sleep(random.uniform(3, 6))
    
    # Generate realistic video analysis results
    fake_percentage = random.uniform(0, 100)
    total_frames = random.randint(100, 1000)
    fake_frames = int((fake_percentage / 100) * total_frames)
    
    return {
        "fake_percentage": round(fake_percentage, 2),
        "total_frames": total_frames,
        "fake_frames": fake_frames,
        "analysis_time": datetime.now().strftime("%H:%M:%S"),
        "file_type": "video"
    }

# HTML Templates with embedded CSS
LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrueLens - Deepfake Detection Login</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #ffdde1, #ee9ca7, #a18cd1, #fbc2eb);
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        
        .login-container {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            border-radius: 25px;
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.15);
            padding: 40px;
            width: 100%;
            max-width: 420px;
            text-align: center;
            animation: fadeIn 1.5s ease-in-out;
        }

        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(20px);}
            to {opacity: 1; transform: translateY(0);}
        }
        
        .logo {
            background: linear-gradient(135deg, #ff6ec7, #a18cd1);
            width: 90px;
            height: 90px;
            border-radius: 25px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            color: white;
            font-size: 2.5rem;
            animation: bounce 2s infinite;
        }

        @keyframes bounce {
            0%, 100% {transform: translateY(0);}
            50% {transform: translateY(-10px);}
        }
        
        h1 {
            color: #333;
            margin-bottom: 5px;
            font-size: 2.2rem;
            font-weight: 700;
        }

        .company {
            color: #ff6ec7;
            font-weight: 800;
            font-size: 1.3rem;
            margin-bottom: 15px;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1rem;
        }
        
        .form-group {
            margin-bottom: 20px;
            text-align: left;
        }
        
        .form-control {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid #e1e5e9;
            border-radius: 15px;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: white;
        }
        
        .form-control:focus {
            outline: none;
            border-color: #ff6ec7;
            box-shadow: 0 0 0 3px rgba(255, 110, 199, 0.2);
        }
        
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #ff6ec7, #a18cd1);
            color: white;
            border: none;
            border-radius: 15px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(255, 110, 199, 0.4);
        }
        
        .alert {
            background: #ffe6ec;
            color: #c33;
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid #ffccd5;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .toggle-link {
            margin-top: 15px;
            font-size: 0.95rem;
            color: #a18cd1;
            cursor: pointer;
            display: inline-block;
            transition: color 0.3s;
        }

        .toggle-link:hover {
            color: #ff6ec7;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">
            <i class="fas fa-shield-alt"></i>
        </div>
        <div class="company">TrueLens Company</div>
        <h1>Deepfake Detection</h1>
        <p class="subtitle">AI-powered media authentication with style ✨</p>
        
        {% if error %}
        <div class="alert">
            <i class="fas fa-exclamation-triangle"></i>
            {{ error }}
        </div>
        {% endif %}
        
        <form method="POST">
            <div class="form-group">
                <input type="text" name="username" class="form-control" placeholder="Username (letters only)" 
                       pattern="[A-Za-z]+" title="Username must contain only letters" required>
            </div>
            <div class="form-group">
                <input type="password" name="password" class="form-control" placeholder="Password (letters & numbers)" 
                       pattern="[A-Za-z0-9]+" title="Password must contain letters and numbers" required>
            </div>
            <button type="submit" class="btn">
                <i class="fas fa-sign-in-alt"></i>
                Login
            </button>
        </form>

        <div class="toggle-link" onclick="window.location.href='/signup'">
            Don’t have an account? Sign up here 🌸
        </div>
    </div>
</body>
</html>
'''


DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deepfake Detection System</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f8fafc;
            min-height: 100vh;
        }
        
        .header {
            background: white;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px 0;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo-section {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logo {
            background: linear-gradient(135deg, #667eea, #764ba2);
            width: 50px;
            height: 50px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5rem;
        }
        
        .logo-text {
            font-size: 1.5rem;
            font-weight: 700;
            color: #333;
        }
        
        .logout-btn {
            background: #e2e8f0;
            color: #64748b;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .logout-btn:hover {
            background: #cbd5e1;
            color: #475569;
        }
        
        .main-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        
        .hero-section {
            text-align: center;
            margin-bottom: 60px;
        }
        
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            color: #1e293b;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .hero-subtitle {
            font-size: 1.25rem;
            color: #64748b;
            max-width: 600px;
            margin: 0 auto;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }
        
        .feature-card {
            background: white;
            border-radius: 20px;
            padding: 40px 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            text-align: center;
            border: 1px solid #e2e8f0;
        }
        
        .feature-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        }
        
        .feature-icon {
            width: 80px;
            height: 80px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 25px;
            font-size: 2rem;
            color: white;
        }
        
        .feature-icon.image {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        }
        
        .feature-icon.video {
            background: linear-gradient(135deg, #ef4444, #dc2626);
        }
        
        .feature-icon.live {
            background: linear-gradient(135deg, #10b981, #059669);
        }
        
        .feature-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 15px;
        }
        
        .feature-description {
            color: #64748b;
            margin-bottom: 30px;
            line-height: 1.6;
        }
        
        .upload-area {
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .upload-area:hover {
            border-color: #667eea;
            background: #f8fafc;
        }
        
        .upload-area.dragover {
            border-color: #667eea;
            background: #eff6ff;
        }
        
        .upload-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }
        
        .upload-icon {
            font-size: 2rem;
            color: #94a3b8;
        }
        
        .upload-text {
            color: #64748b;
            font-weight: 500;
        }
        
        .file-input {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }
        
        .analyze-btn {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .analyze-btn.image {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            color: white;
        }
        
        .analyze-btn.video {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
        }
        
        .analyze-btn.live {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
        }
        
        .analyze-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }
        
        .analyze-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        
        .loading-content {
            background: white;
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            max-width: 400px;
            width: 90%;
        }
        
        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid #e2e8f0;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 20px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s ease;
            animation: progress 3s ease-in-out;
        }
        
        @keyframes progress {
            0% { width: 0%; }
            50% { width: 60%; }
            100% { width: 100%; }
        }
        
        @media (max-width: 768px) {
            .features-grid {
                grid-template-columns: 1fr;
            }
            
            .hero-title {
                font-size: 2rem;
            }
            
            .header-content {
                flex-direction: column;
                gap: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="logo-section">
                <div class="logo">
                    <i class="fas fa-shield-alt"></i>
                </div>
                <div class="logo-text">Deepfake Detection System</div>
            </div>
            <a href="/logout" class="logout-btn">
                <i class="fas fa-sign-out-alt"></i>
                Logout
            </a>
        </div>
    </div>
    
    <div class="main-content">
        <div class="hero-section">
            <h1 class="hero-title">AI-Powered Media Authentication</h1>
            <p class="hero-subtitle">Analyze images and videos for potential deepfake content using advanced machine learning algorithms</p>
        </div>
        
        <div class="features-grid">
            <!-- Image Analysis -->
            <div class="feature-card">
                <div class="feature-icon image">
                    <i class="fas fa-image"></i>
                </div>
                <h3 class="feature-title">Image Analysis</h3>
                <p class="feature-description">Upload an image to detect potential deepfake manipulation using advanced neural networks</p>
                
                <form id="imageForm" action="/upload-image" method="post" enctype="multipart/form-data">
                    <div class="upload-area" onclick="document.getElementById('imageFile').click()">
                        <input type="file" id="imageFile" name="file" class="file-input" accept="image/*" required>
                        <div class="upload-content">
                            <i class="fas fa-cloud-upload-alt upload-icon"></i>
                            <div class="upload-text">Click to upload image</div>
                            <small style="color: #94a3b8;">PNG, JPG, JPEG, AVIF supported</small>
                        </div>
                    </div>
                    <button type="submit" class="analyze-btn image">
                        <i class="fas fa-search"></i>
                        Analyze Image
                    </button>
                </form>
            </div>
            
            <!-- Video Analysis -->
            <div class="feature-card">
                <div class="feature-icon video">
                    <i class="fas fa-video"></i>
                </div>
                <h3 class="feature-title">Video Analysis</h3>
                <p class="feature-description">Analyze video files frame-by-frame for comprehensive deepfake detection</p>
                
                <form id="videoForm" action="/upload-video" method="post" enctype="multipart/form-data">
                    <div class="upload-area" onclick="document.getElementById('videoFile').click()">
                        <input type="file" id="videoFile" name="file" class="file-input" accept="video/*" required>
                        <div class="upload-content">
                            <i class="fas fa-film upload-icon"></i>
                            <div class="upload-text">Click to upload video</div>
                            <small style="color: #94a3b8;">MP4, AVI, MOV supported</small>
                        </div>
                    </div>
                    <button type="submit" class="analyze-btn video">
                        <i class="fas fa-play"></i>
                        Analyze Video
                    </button>
                </form>
            </div>
            
            <!-- Live Detection -->
            <div class="feature-card">
                <div class="feature-icon live">
                    <i class="fas fa-camera"></i>
                </div>
                <h3 class="feature-title">Live Detection</h3>
                <p class="feature-description">Real-time webcam analysis for live deepfake detection and monitoring</p>
                
                <div style="margin-bottom: 20px; padding: 20px; background: #f1f5f9; border-radius: 12px;">
                    <i class="fas fa-info-circle" style="color: #3b82f6; margin-right: 8px;"></i>
                    <span style="color: #475569;">Real-time webcam analysis</span>
                </div>
                
                <a href="/live-video" class="analyze-btn live" style="text-decoration: none;">
                    <i class="fas fa-play"></i>
                    Start Live Detection
                </a>
            </div>
        </div>
    </div>
    
    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-content">
            <div class="spinner"></div>
            <h3 style="color: #1e293b; margin-bottom: 10px;">Analyzing Content</h3>
            <p style="color: #64748b; margin-bottom: 0;">Our AI is processing your file using advanced deepfake detection algorithms...</p>
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Show loading overlay on form submission
        document.getElementById('imageForm').addEventListener('submit', function() {
            document.getElementById('loadingOverlay').style.display = 'flex';
        });
        
        document.getElementById('videoForm').addEventListener('submit', function() {
            document.getElementById('loadingOverlay').style.display = 'flex';
        });
        
        // File upload feedback
        document.getElementById('imageFile').addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name;
            if (fileName) {
                const uploadText = this.parentElement.querySelector('.upload-text');
                uploadText.textContent = fileName;
                uploadText.style.color = '#667eea';
            }
        });
        
        document.getElementById('videoFile').addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name;
            if (fileName) {
                const uploadText = this.parentElement.querySelector('.upload-text');
                uploadText.textContent = fileName;
                uploadText.style.color = '#ef4444';
            }
        });
        
        // Drag and drop functionality
        function setupDragDrop(uploadArea, fileInput) {
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                this.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                this.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                this.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    const event = new Event('change', { bubbles: true });
                    fileInput.dispatchEvent(event);
                }
            });
        }
        
        // Setup drag and drop for both upload areas
        const imageUploadArea = document.querySelector('#imageForm .upload-area');
        const videoUploadArea = document.querySelector('#videoForm .upload-area');
        const imageFileInput = document.getElementById('imageFile');
        const videoFileInput = document.getElementById('videoFile');
        
        setupDragDrop(imageUploadArea, imageFileInput);
        setupDragDrop(videoUploadArea, videoFileInput);
    </script>
</body>
</html>
'''

IMAGE_RESULT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Analysis Result</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f8fafc;
            min-height: 100vh;
            padding: 20px;
        }
        
        .result-container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            animation: slideUp 0.6s ease-out;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result-header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .result-icon {
            font-size: 4rem;
            margin-bottom: 20px;
            animation: bounce 0.8s ease-out;
        }
        
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {
                transform: translateY(0);
            }
            40% {
                transform: translateY(-10px);
            }
            60% {
                transform: translateY(-5px);
            }
        }
        
        .result-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .result-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .result-content {
            padding: 40px;
        }
        
        .classification-badge {
            display: inline-block;
            padding: 15px 30px;
            border-radius: 50px;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 30px;
            text-align: center;
            width: 100%;
        }
        
        .classification-badge.real {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
        }
        
        .classification-badge.fake {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
        }
        
        .image-preview {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .preview-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease;
        }
        
        .preview-image:hover {
            transform: scale(1.02);
        }
        
        .analysis-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .detail-card {
            background: #f8fafc;
            padding: 25px;
            border-radius: 16px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }
        
        .detail-icon {
            font-size: 2rem;
            margin-bottom: 15px;
            color: #667eea;
        }
        
        .detail-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 10px;
        }
        
        .detail-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #667eea;
        }
        
        .confidence-bar {
            background: #e2e8f0;
            height: 12px;
            border-radius: 6px;
            overflow: hidden;
            margin: 15px 0;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 1s ease-out;
            animation: fillBar 1.5s ease-out;
        }
        
        .confidence-fill.real {
            background: linear-gradient(135deg, #10b981, #059669);
        }
        
        .confidence-fill.fake {
            background: linear-gradient(135deg, #ef4444, #dc2626);
        }
        
        @keyframes fillBar {
            from { width: 0%; }
        }
        
        .action-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 15px 30px;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        
        .btn-secondary {
            background: #e2e8f0;
            color: #64748b;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }
        
        @media (max-width: 768px) {
            .result-container {
                margin: 10px;
                border-radius: 16px;
            }
            
            .result-header {
                padding: 30px 20px;
            }
            
            .result-content {
                padding: 30px 20px;
            }
            
            .action-buttons {
                flex-direction: column;
            }
            
            .btn {
                width: 100%;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="result-container">
        <div class="result-header">
            <div class="result-icon">
                <i class="fas {{ 'fa-times-circle' if result.result == 'Fake' else 'fa-check-circle' }}"></i>
            </div>
            <h1 class="result-title">Image Analysis Complete</h1>
            <p class="result-subtitle">AI-powered deepfake detection results</p>
        </div>
        
        <div class="result-content">
            <div class="classification-badge {{ 'fake' if result.result == 'Fake' else 'real' }}">
                Classification: {{ result.result }}
            </div>
            
            <div class="image-preview">
                <img src="{{ url_for('static', filename='uploads/' + filename) }}" alt="Analyzed Image" class="preview-image">
            </div>
            
            <div class="analysis-details">
                <div class="detail-card">
                    <div class="detail-icon">
                        <i class="fas fa-percentage"></i>
                    </div>
                    <div class="detail-title">Confidence Level</div>
                    <div class="detail-value">{{ result.confidence }}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill {{ 'fake' if result.result == 'Fake' else 'real' }}" 
                             style="width: {{ result.confidence }}%"></div>
                    </div>
                </div>
                
                <div class="detail-card">
                    <div class="detail-icon">
                        <i class="fas fa-clock"></i>
                    </div>
                    <div class="detail-title">Analysis Time</div>
                    <div class="detail-value">{{ result.analysis_time }}</div>
                    <p style="color: #64748b; margin-top: 10px; font-size: 0.9rem;">
                        Deep neural network analysis
                    </p>
                </div>
                
                <div class="detail-card">
                    <div class="detail-icon">
                        <i class="fas fa-file-image"></i>
                    </div>
                    <div class="detail-title">File Type</div>
                    <div class="detail-value">{{ result.file_type.upper() }}</div>
                    <p style="color: #64748b; margin-top: 10px; font-size: 0.9rem;">
                        {{ filename }}
                    </p>
                </div>
            </div>
            
            <div class="action-buttons">
                <a href="/home" class="btn btn-primary">
                    <i class="fas fa-redo"></i>
                    Analyze Another
                </a>
                <a href="/home" class="btn btn-secondary">
                    <i class="fas fa-home"></i>
                    Back to Dashboard
                </a>
            </div>
        </div>
    </div>
</body>
</html>
'''

VIDEO_RESULT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Analysis Result</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f8fafc;
            min-height: 100vh;
            padding: 20px;
        }
        
        .result-container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            animation: slideUp 0.6s ease-out;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result-header {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .result-icon {
            font-size: 4rem;
            margin-bottom: 20px;
            animation: bounce 0.8s ease-out;
        }
        
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {
                transform: translateY(0);
            }
            40% {
                transform: translateY(-10px);
            }
            60% {
                transform: translateY(-5px);
            }
        }
        
        .result-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .result-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .result-content {
            padding: 40px;
        }
        
        .percentage-display {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .percentage-circle {
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: conic-gradient(
                #ef4444 0deg {{ result.fake_percentage * 3.6 }}deg,
                #e2e8f0 {{ result.fake_percentage * 3.6 }}deg 360deg
            );
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            position: relative;
            animation: rotateIn 1s ease-out;
        }
        
        @keyframes rotateIn {
            from {
                transform: rotate(-90deg);
                opacity: 0;
            }
            to {
                transform: rotate(0deg);
                opacity: 1;
            }
        }
        
        .percentage-inner {
            width: 160px;
            height: 160px;
            background: white;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .percentage-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: #ef4444;
        }
        
        .percentage-label {
            font-size: 0.9rem;
            color: #64748b;
            font-weight: 600;
        }
        
        .analysis-summary {
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            padding: 25px;
            border-radius: 16px;
            margin-bottom: 30px;
            text-align: center;
            border: 1px solid #fca5a5;
        }
        
        .summary-text {
            font-size: 1.25rem;
            color: #991b1b;
            font-weight: 600;
        }
        
        .video-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background: #f8fafc;
            padding: 25px;
            border-radius: 16px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }
        
        .stat-icon {
            font-size: 2rem;
            margin-bottom: 15px;
            color: #ef4444;
        }
        
        .stat-title {
            font-size: 1rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 10px;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #ef4444;
        }
        
        .progress-section {
            margin-bottom: 40px;
        }
        
        .progress-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .progress-bar {
            background: #e2e8f0;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #ef4444, #dc2626);
            border-radius: 10px;
            transition: width 2s ease-out;
            animation: fillProgress 2s ease-out;
        }
        
        @keyframes fillProgress {
            from { width: 0%; }
        }
        
        .progress-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-weight: 600;
            font-size: 0.9rem;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
        }
        
        .action-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 15px 30px;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        
        .btn-secondary {
            background: #e2e8f0;
            color: #64748b;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }
        
        @media (max-width: 768px) {
            .result-container {
                margin: 10px;
                border-radius: 16px;
            }
            
            .result-header {
                padding: 30px 20px;
            }
            
            .result-content {
                padding: 30px 20px;
            }
            
            .percentage-circle {
                width: 150px;
                height: 150px;
            }
            
            .percentage-inner {
                width: 120px;
                height: 120px;
            }
            
            .percentage-value {
                font-size: 2rem;
            }
            
            .action-buttons {
                flex-direction: column;
            }
            
            .btn {
                width: 100%;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="result-container">
        <div class="result-header">
            <div class="result-icon">
                <i class="fas fa-video"></i>
            </div>
            <h1 class="result-title">Video Analysis Complete</h1>
            <p class="result-subtitle">Frame-by-frame deepfake detection results</p>
        </div>
        
        <div class="result-content">
            <div class="percentage-display">
                <div class="percentage-circle">
                    <div class="percentage-inner">
                        <div class="percentage-value">{{ "%.1f"|format(result.fake_percentage) }}%</div>
                        <div class="percentage-label">FAKE CONTENT</div>
                    </div>
                </div>
            </div>
            
            <div class="analysis-summary">
                <div class="summary-text">
                    This video contains <strong>{{ "%.1f"|format(result.fake_percentage) }}%</strong> potential fake content
                </div>
            </div>
            
            <div class="progress-section">
                <div class="progress-title">Fake Content Distribution</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ result.fake_percentage }}%"></div>
                    <div class="progress-text">{{ result.fake_frames }} of {{ result.total_frames }} frames</div>
                </div>
            </div>
            
            <div class="video-stats">
                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-film"></i>
                    </div>
                    <div class="stat-title">Total Frames</div>
                    <div class="stat-value">{{ result.total_frames }}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-exclamation-triangle"></i>
                    </div>
                    <div class="stat-title">Fake Frames</div>
                    <div class="stat-value">{{ result.fake_frames }}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-clock"></i>
                    </div>
                    <div class="stat-title">Analysis Time</div>
                    <div class="stat-value">{{ result.analysis_time }}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-file-video"></i>
                    </div>
                    <div class="stat-title">File Type</div>
                    <div class="stat-value">{{ result.file_type.upper() }}</div>
                </div>
            </div>
            
            <div class="action-buttons">
                <a href="/home" class="btn btn-primary">
                    <i class="fas fa-redo"></i>
                    Analyze Another
                </a>
                <a href="/home" class="btn btn-secondary">
                    <i class="fas fa-home"></i>
                    Back to Dashboard
                </a>
            </div>
        </div>
    </div>
</body>
</html>
'''

# Routes
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == "admin" and password == "admin":
            session['user'] = username
            return redirect(url_for("home"))
        else:
            return render_template_string(LOGIN_TEMPLATE, error="Invalid credentials. Use admin/admin")
    return render_template_string(LOGIN_TEMPLATE)

@app.route("/home")
def home():
    if 'user' not in session:
        return redirect(url_for("login"))
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route("/upload-image", methods=["POST"])
def upload_image():
    if 'user' not in session:
        return redirect(url_for("login"))
        
    if "file" not in request.files:
        return redirect(url_for("home"))

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    result = mock_predict_image(filepath)
    
    return render_template_string(IMAGE_RESULT_TEMPLATE, result=result, filename=filename)

@app.route("/upload-video", methods=["POST"])
def upload_video():
    if 'user' not in session:
        return redirect(url_for("login"))
        
    if "file" not in request.files:
        return redirect(url_for("home"))

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    result = mock_predict_video(filepath)
    result['filename'] = filename
    
    return render_template_string(VIDEO_RESULT_TEMPLATE, result=result, filename=filename)

@app.route("/live-video")
def live_video():
    if 'user' not in session:
        return redirect(url_for("login"))
    
    # For demo purposes, return a simple page explaining live detection
    live_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live Detection</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #10b981, #059669);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .live-container {
                background: white;
                border-radius: 24px;
                padding: 40px;
                text-align: center;
                max-width: 500px;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.2);
            }
            .live-icon {
                font-size: 4rem;
                color: #10b981;
                margin-bottom: 20px;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }
            h1 {
                color: #1e293b;
                margin-bottom: 20px;
                font-size: 2rem;
            }
            p {
                color: #64748b;
                margin-bottom: 30px;
                line-height: 1.6;
            }
            .btn {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 12px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 10px;
                transition: all 0.3s ease;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
            }
        </style>
    </head>
    <body>
        <div class="live-container">
            <div class="live-icon">
                <i class="fas fa-camera"></i>
            </div>
            <h1>Live Detection</h1>
            <p>Live webcam detection would be implemented here using WebRTC and real-time video processing. This feature requires camera permissions and continuous frame analysis.</p>
            <p><strong>Note:</strong> This is a demo version. In a production environment, this would connect to your webcam and perform real-time deepfake detection.</p>
            <a href="/home" class="btn">
                <i class="fas fa-arrow-left"></i>
                Back to Dashboard
            </a>
        </div>
    </body>
    </html>
    '''
    return render_template_string(live_template)

@app.route("/logout")
def logout():
    session.pop('user', None)
    return redirect(url_for("login"))

# Static file serving
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return app.send_static_file(f'uploads/{filename}')

if _name_ == "_main_":
    print("🚀 Starting Deepfake Detection System...")
    print("📱 Access the application at: http://localhost:5000")
    print("🔐 Login credentials: admin / admin")
    print("📁 Upload folder:", UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=5000)
