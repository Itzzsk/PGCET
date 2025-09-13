from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os
import requests
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Google Drive direct download URL for your ML model
MODEL_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1KaTsdYcRwxSOJTyfhc8FHyz-HYAXSb-a"

def download_from_google_drive(url, filename):
    """Download file from Google Drive if it doesn't exist locally"""
    if os.path.exists(filename):
        print(f"✅ {filename} already exists locally")
        return True
    
    try:
        print(f"📥 Downloading {filename} from Google Drive...")
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Show progress for large files
                        if total_size > 0 and downloaded % (1024*1024) == 0:  # Every 1MB
                            percent = (downloaded / total_size) * 100
                            print(f"📊 Progress: {percent:.1f}%")
            
            print(f"✅ Successfully downloaded {filename}")
            return True
        else:
            print(f"❌ Failed to download {filename}. Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading {filename}: {str(e)}")
        return False

# Load the ML model with Google Drive integration
try:
    # Download model from Google Drive if needed
    model_downloaded = download_from_google_drive(MODEL_DOWNLOAD_URL, 'advanced_pgcet_model.pkl')
    
    if model_downloaded:
        from advanced_ml_predictor import AdvancedPGCETPredictor
        predictor = AdvancedPGCETPredictor.load_models('advanced_pgcet_model.pkl', 'combined_pgcet_data.json')
        print("✅ ML models loaded successfully from Google Drive")
    else:
        predictor = None
        print("⚠️ Using fallback mode - ML model download failed")
        
except Exception as e:
    print(f"⚠️ Using fallback mode: {e}")
    predictor = None

# Rest of your Flask app code continues here...


# Mobile-First HTML Template with FontAwesome
MOBILE_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>PGCET Cutoff Finder</title>
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
    
    <!-- FontAwesome CDN -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
      <script async src="https://www.googletagmanager.com/gtag/js?id=G-FB4PJCD0NM"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-FB4PJCD0NM');
</script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 10px;
        }
        .container { max-width: 100%; }
        
        /* Navigation Bar */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255,255,255,0.95);
            padding: 15px 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .nav-logo {
            display: flex;
            align-items: center;
            font-size: 1.2rem;
            font-weight: 700;
            color: #2c3e50;
        }
        .nav-logo i {
            margin-right: 8px;
            color: #667eea;
        }
        .nav-menu {
            position: relative;
        }
        .menu-toggle {
            background: none;
            border: none;
            font-size: 1.3rem;
            color: #667eea;
            cursor: pointer;
            padding: 8px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .menu-toggle:hover {
            background: #f8f9fa;
        }
        
        /* Dropdown Menu */
        .dropdown-menu {
            position: absolute;
            top: 100%;
            right: 0;
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            min-width: 200px;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transform: translateY(-10px);
            transition: all 0.3s ease;
        }
        .dropdown-menu.show {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
        }
        .dropdown-item {
            display: flex;
            align-items: center;
            padding: 15px 20px;
            color: #2c3e50;
            text-decoration: none;
            border-bottom: 1px solid #f1f2f6;
            transition: background 0.2s ease;
        }
        .dropdown-item:hover {
            background: #f8f9fa;
        }
        .dropdown-item:last-child {
            border-bottom: none;
        }
        .dropdown-item i {
            margin-right: 12px;
            width: 20px;
            color: #667eea;
        }
        
        /* Header */
        .header { 
            text-align: center; 
            color: white; 
            margin-bottom: 20px;
            padding: 20px 10px;
        }
        .header h1 { 
            font-size: 1.8rem; 
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .header h1 i {
            margin-right: 10px;
        }
        .header p { 
            font-size: 0.9rem; 
            opacity: 0.9;
        }
        
        /* Search Card */
        .search-card { 
            background: white; 
            border-radius: 15px; 
            padding: 20px; 
            margin-bottom: 20px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .form-group { 
            margin-bottom: 15px; 
        }
        .form-group label { 
            display: block; 
            margin-bottom: 8px; 
            font-weight: 600; 
            color: #2c3e50;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
        }
        .form-group label i {
            margin-right: 8px;
            color: #667eea;
            width: 16px;
        }
        .form-group input, .form-group select { 
            width: 100%; 
            padding: 15px; 
            border: 2px solid #ecf0f1; 
            border-radius: 10px; 
            font-size: 16px;
            -webkit-appearance: none;
        }
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        /* City Preference */
        .city-preference {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .city-preference h4 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
        }
        .city-preference h4 i {
            margin-right: 8px;
            color: #667eea;
        }
        
        .search-btn { 
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            color: white; 
            padding: 18px; 
            border: none; 
            border-radius: 12px; 
            font-size: 16px; 
            font-weight: 600;
            cursor: pointer; 
            width: 100%;
            margin-top: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .search-btn i {
            margin-right: 10px;
        }
        .search-btn:active { 
            transform: scale(0.98);
        }
        
        /* Results */
        .results-header { 
            background: white; 
            padding: 15px; 
            border-radius: 12px; 
            margin-bottom: 15px; 
            text-align: center;
        }
        .results-header h2 {
            font-size: 1.3rem;
            color: #2c3e50;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .results-header h2 i {
            margin-right: 10px;
            color: #27ae60;
        }
        .results-header p {
            font-size: 0.9rem;
            color: #7f8c8d;
        }
        
        /* College Cards */
        .college-list {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .college-card { 
            background: white; 
            border-radius: 12px; 
            padding: 20px; 
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            position: relative;
        }
        .college-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            border-radius: 12px 12px 0 0;
        }
        
        .college-rank { 
            position: absolute; 
            top: 15px; 
            right: 15px; 
            background: #667eea; 
            color: white; 
            width: 35px; 
            height: 35px; 
            border-radius: 50%; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            font-weight: bold;
            font-size: 0.9rem;
        }
        
        .college-name { 
            font-size: 1.1rem; 
            font-weight: 600; 
            color: #2c3e50; 
            margin-bottom: 8px;
            padding-right: 45px;
            line-height: 1.3;
        }
        .college-location { 
            color: #7f8c8d; 
            margin-bottom: 15px;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
        }
        .college-location i {
            margin-right: 6px;
            color: #e74c3c;
        }
        
        /* MOBILE CUTOFF DISPLAY */
        .cutoff-main { 
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            text-align: center;
        }
        .cutoff-rank { 
            font-size: 2.2rem; 
            font-weight: bold; 
            margin-bottom: 5px;
        }
        .cutoff-label { 
            font-size: 0.85rem; 
            opacity: 0.9;
        }
        
        /* Rank Comparison */
        .rank-comparison { 
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .your-rank { 
            font-size: 0.9rem; 
            color: #2c3e50; 
        }
        .your-rank-value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #e74c3c;
        }
        .rank-difference { 
            text-align: right;
            font-size: 0.9rem;
        }
        .difference-value {
            font-size: 1.1rem;
            font-weight: bold;
        }
        .difference-positive { color: #27ae60; }
        .difference-negative { color: #e74c3c; }
        
        /* Stats Mobile */
        .stats-mobile { 
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }
        .stat-mobile { 
            text-align: center; 
            flex: 1;
            padding: 12px 8px;
            background: #f8f9fa;
            border-radius: 8px;
            margin: 0 3px;
        }
        .stat-mobile:first-child { margin-left: 0; }
        .stat-mobile:last-child { margin-right: 0; }
        .stat-label-mobile { 
            font-size: 0.75rem; 
            color: #7f8c8d; 
            display: block;
            margin-bottom: 4px;
        }
        .stat-value-mobile { 
            font-size: 1rem; 
            font-weight: 600; 
            color: #2c3e50;
        }
        
        .safety-badge { 
            display: inline-block;
            padding: 8px 16px; 
            border-radius: 20px; 
            font-size: 0.8rem; 
            font-weight: 600;
            color: white;
        }
        .very-safe { background: #27ae60; }
        .safe { background: #2ecc71; }
        .moderate { background: #f39c12; }
        .competitive { background: #e67e22; }
        .reach { background: #e74c3c; }
        
        .round-info {
            background: #e8f4f8;
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 0.75rem;
            color: #2c3e50;
            display: inline-block;
            margin-left: 10px;
        }
        
        /* Cutoff Toggle */
        .cutoff-toggle {
            background: none;
            border: 1px solid #667eea;
            color: #667eea;
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 0.8rem;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .cutoff-toggle i {
            margin-right: 8px;
        }
        .cutoff-toggle:active {
            background: #667eea;
            color: white;
        }
        
        .cutoff-details {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-top: 10px;
        }
        .cutoff-categories {
            font-size: 0.85rem;
        }
        .cutoff-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        .cutoff-row:last-child {
            border-bottom: none;
        }
        
        /* Loading & No Results */
        .loading { 
            text-align: center; 
            color: white; 
            padding: 40px 20px;
        }
        .loading h2 {
            font-size: 1.3rem;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .loading h2 i {
            margin-right: 10px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .no-results {
            text-align: center;
            padding: 40px 20px;
            background: white;
            border-radius: 12px;
        }
        .no-results h2 {
            font-size: 1.3rem;
            color: #2c3e50;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .no-results h2 i {
            margin-right: 10px;
            color: #e74c3c;
        }
        .no-results p {
            color: #7f8c8d;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        
        /* About Modal */
        .modal {
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }
        .modal-content {
            background-color: white;
            margin: 5% auto;
            padding: 20px;
            border-radius: 15px;
            width: 90%;
            max-width: 500px;
            max-height: 80vh;
            overflow-y: auto;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #ecf0f1;
        }
        .modal-header h2 {
            font-size: 1.3rem;
            color: #2c3e50;
            display: flex;
            align-items: center;
        }
        .modal-header h2 i {
            margin-right: 10px;
            color: #667eea;
        }
        .close {
            background: none;
            border: none;
            font-size: 1.5rem;
            color: #7f8c8d;
            cursor: pointer;
            padding: 5px;
            border-radius: 5px;
        }
        .close:hover {
            background: #f8f9fa;
        }
        
        .about-section {
            margin-bottom: 20px;
        }
        .about-section h3 {
            font-size: 1.1rem;
            color: #2c3e50;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        .about-section h3 i {
            margin-right: 8px;
            color: #667eea;
        }
        .about-section p {
            color: #7f8c8d;
            line-height: 1.5;
            margin-bottom: 10px;
        }
        .feature-list {
            list-style: none;
            padding: 0;
        }
        .feature-list li {
            padding: 8px 0;
            display: flex;
            align-items: center;
            color: #2c3e50;
        }
        .feature-list li i {
            margin-right: 10px;
            color: #27ae60;
            width: 20px;
        }
        
        /* Data Status */
        .data-status {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }
        .data-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .data-info span {
            font-size: 0.85rem;
            color: #7f8c8d;
        }
        .data-info strong {
            color: #2c3e50;
        }
        
        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 0.9rem;
            cursor: pointer;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 10px;
        }
        .refresh-btn i {
            margin-right: 8px;
        }
        .refresh-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .developer-info {
    background: #f5f5f5;
    border: 2px solid #000;
    padding: 20px;
    margin: 20px 0;
}

/* Developer Info - Centered with LinkedIn Original Color */
.developer-info {
    background: #f5f5f5;
    padding: 25px;
    margin: 25px 0;
    text-align: center;
}

.developer-info h4 {
    font-size: 1rem;
    color: #000;
    margin-bottom: 15px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.developer-info h4 i {
    margin-right: 8px;
    color: #000;
}

.developer-info p {
    color: #333;
    margin-bottom: 15px;
    font-weight: 500;
    font-size: 1rem;
}

.developer-info p strong {
    color: #000;
    font-weight: 700;
    font-size: 1.1rem;
}

/* LinkedIn Icon with Original Blue Color */
.linkedin-link {
    display: inline-block;
    color: #0077B5;
    padding: 12px;
    text-decoration: none;
    transition: all 0.2s ease;
}

.linkedin-link i {
    font-size: 1.2rem;
}

.linkedin-link:hover {
    color: #005885;
}


    </style>
</head>
<body>
    <div class="container">
        <!-- Navigation Bar -->
        <div class="navbar">
            <div class="nav-logo">
                <i class="fas fa-graduation-cap"></i>
               PGCET MCA Guide
            </div>
            <div class="nav-menu">
                <button class="menu-toggle" onclick="toggleMenu()">
                    <i class="fas fa-bars"></i>
                </button>
                <div class="dropdown-menu" id="dropdownMenu">
                    <a href="#" class="dropdown-item" onclick="showAbout()">
                        <i class="fas fa-info-circle"></i>
                        About
                    </a>
                    <a href="#" class="dropdown-item" onclick="refreshData()">
                        <i class="fas fa-sync-alt"></i>
                        Refresh Data
                    </a>
                    <a href="#" class="dropdown-item" onclick="clearResults()">
                        <i class="fas fa-eraser"></i>
                        Clear Results
                    </a>
                </div>
            </div>
        </div>
        
        <div class="header">
            <h1><i class="fas fa-bullseye"></i>PGCET Cutoff Finder</h1>
            <p>Find colleges where you meet the cutoff</p>
        </div>
        
        <div class="search-card">
            <div class="form-group">
                <label><i class="fas fa-hashtag"></i>Your PGCET Rank:</label>
                <input type="number" id="rank" placeholder="Enter your rank (e.g., 3500)">
            </div>
            <div class="form-group">
                <label><i class="fas fa-users"></i>Category:</label>
                <select id="category">
                    <option value="GM">General Merit (GM)</option>
                    <option value="1G">Category 1G</option>
                    <option value="2AG">Category 2AG</option>
                    <option value="2BG">Category 2BG</option>
                    <option value="3AG">Category 3AG</option>
                    <option value="3BG">Category 3BG</option>
                    <option value="SCG">SC General (SCG)</option>
                    <option value="SCH">SC Hyderabad-Karnataka (SCH)</option>
                    <option value="STG">ST General (STG)</option>
                    <option value="STH">ST Hyderabad-Karnataka (STH)</option>
                </select>
            </div>
            
            <div class="city-preference">
                <h4><i class="fas fa-map-marker-alt"></i>Preferred City (Optional)</h4>
                <select id="preferredCity">
                    <option value="">Any City</option>
                    <option value="BANGALORE">Bangalore</option>
                    <option value="MYSORE">Mysore</option>
                    <option value="HUBLI">Hubli</option>
                    <option value="MANGALORE">Mangalore</option>
                </select>
            </div>
            
            <button class="search-btn" onclick="searchColleges()">
                <i class="fas fa-search"></i>Find My Colleges
            </button>
        </div>
        
        <div id="results"></div>
    </div>
    
    <!-- About Modal -->
    <div id="aboutModal" class="modal">
    <div class="modal-content">
        <div class="modal-header">
            <h2><i class="fas fa-info-circle"></i>About PGCET Finder</h2>
            <button class="close" onclick="closeAbout()">&times;</button>
        </div>
        
        <div class="about-section">
            <h3><i class="fas fa-bullseye"></i>What This Does</h3>
            <p>Helps Karnataka PGCET students quickly find colleges where they can get admission based on their rank and category. No more searching through hundreds of PDF pages!</p>
        </div>
        
       <div class="about-section">
    <h3><i class="fas fa-rocket"></i>Key Features</h3>
    <ul class="feature-list">
        <li><i class="fas fa-search"></i>Instant college search by rank</li>
        <li><i class="fas fa-chart-line"></i>Clear cutoff comparison</li>
        <li><i class="fas fa-heart"></i>Completely free to use</li>
    </ul>
</div>

<div class="developer-info">
    <h4><i class="fas fa-code"></i>Created By</h4>
    <p><strong>Skanda Umesh</strong></p>
    <a href="https://www.linkedin.com/in/skanda-umesh-88b16432b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank" class="linkedin-link">
        <i class="fa-brands fa-linkedin"></i>
    </a>
</div>



    </div>
</div>

    
    <script>
        let currentStudentRank = 0;
        let currentCategory = '';
        
        // Initialize app
        document.addEventListener('DOMContentLoaded', function() {
            loadDataStatus();
        });
        
        // Menu Toggle
        function toggleMenu() {
            const menu = document.getElementById('dropdownMenu');
            menu.classList.toggle('show');
        }
        
        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            const menu = document.getElementById('dropdownMenu');
            const toggle = document.querySelector('.menu-toggle');
            
            if (!menu.contains(event.target) && !toggle.contains(event.target)) {
                menu.classList.remove('show');
            }
        });
        
        // About Modal Functions
        function showAbout() {
            document.getElementById('aboutModal').style.display = 'block';
            document.getElementById('dropdownMenu').classList.remove('show');
            loadDataStatus();
        }
        
        function closeAbout() {
            document.getElementById('aboutModal').style.display = 'none';
        }
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('aboutModal');
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
        
        // Load Data Status
        async function loadDataStatus() {
            try {
                const response = await fetch('/api/data-status');
                const data = await response.json();
                
                document.getElementById('lastUpdated').textContent = data.last_updated || 'Unknown';
                document.getElementById('totalColleges').textContent = data.total_colleges || 'Unknown';
            } catch (error) {
                document.getElementById('lastUpdated').textContent = 'Error loading';
                document.getElementById('totalColleges').textContent = 'Error loading';
            }
        }
        
        // Refresh Data
        async function refreshData() {
            const refreshBtn = document.getElementById('refreshBtn');
            const originalText = refreshBtn.innerHTML;
            
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>Refreshing...';
            
            try {
                const response = await fetch('/api/refresh-data', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    alert('✅ Data refreshed successfully!');
                    loadDataStatus();
                } else {
                    alert('❌ Failed to refresh data: ' + data.error);
                }
            } catch (error) {
                alert('❌ Error refreshing data: ' + error.message);
            } finally {
                refreshBtn.disabled = false;
                refreshBtn.innerHTML = originalText;
            }
            
            document.getElementById('dropdownMenu').classList.remove('show');
        }
        
        // Clear Results
        function clearResults() {
            document.getElementById('results').innerHTML = '';
            document.getElementById('rank').value = '';
            document.getElementById('category').value = 'GM';
            document.getElementById('preferredCity').value = '';
            document.getElementById('dropdownMenu').classList.remove('show');
        }
        
        async function searchColleges() {
            const rank = document.getElementById('rank').value;
            const category = document.getElementById('category').value;
            const preferredCity = document.getElementById('preferredCity').value;
            
            if (!rank) {
                alert('Please enter your PGCET rank');
                return;
            }
            
            currentStudentRank = parseInt(rank);
            currentCategory = category;
            
            // Show loading
            document.getElementById('results').innerHTML = `
                <div class="loading">
                    <h2><i class="fas fa-spinner"></i>Finding colleges...</h2>
                    <p>Checking cutoff requirements</p>
                </div>
            `;
            
            try {
                const response = await fetch('/api/predict-mobile', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        rank: currentStudentRank,
                        category: category,
                        preferences: {
                            preferred_city: preferredCity
                        }
                    })
                });
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                alert('Error: ' + error.message);
                document.getElementById('results').innerHTML = '';
            }
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            if (data.colleges && data.colleges.length > 0) {
                let html = `
                    <div class="results-header">
                        <h2><i class="fas fa-check-circle"></i>${data.colleges.length} colleges found!</h2>
                        <p>Rank ${data.student_rank} • ${data.category}</p>
                    </div>
                    <div class="college-list">
                `;
                
                data.colleges.forEach((college, index) => {
                    const safetyClass = college.safety_level.toLowerCase().replace(' ', '-');
                    const probability = (college.admission_probability * 100).toFixed(0);
                    
                    // Calculate rank difference
                    const rankDiff = college.cutoff_rank - currentStudentRank;
                    const diffClass = rankDiff > 0 ? 'difference-positive' : 'difference-negative';
                    const diffIcon = rankDiff > 0 ? '<i class="fas fa-check-circle"></i>' : '<i class="fas fa-exclamation-triangle"></i>';
                    const diffText = rankDiff > 0 ? 
                        `${rankDiff} ranks safe` : 
                        `${Math.abs(rankDiff)} behind`;
                    
                    html += `
                        <div class="college-card">
                            <div class="college-rank">${index + 1}</div>
                            <div class="college-name">${college.college_name}</div>
                            <div class="college-location"><i class="fas fa-map-marker-alt"></i>${college.city || college.location}</div>
                            
                            <!-- MAIN CUTOFF -->
                            <div class="cutoff-main">
                                <div class="cutoff-rank">${college.cutoff_rank}</div>
                                <div class="cutoff-label">Cutoff Rank • ${currentCategory}</div>
                            </div>
                            
                            <!-- RANK COMPARISON -->
                            <div class="rank-comparison">
                                <div class="your-rank">
                                    Your Rank
                                    <div class="your-rank-value">${currentStudentRank}</div>
                                </div>
                                <div class="rank-difference">
                                    ${diffIcon}
                                    <div class="difference-value ${diffClass}">${diffText}</div>
                                </div>
                            </div>
                            
                            <!-- MOBILE STATS -->
                            <div class="stats-mobile">
                                <div class="stat-mobile">
                                    <span class="stat-label-mobile">Probability</span>
                                    <div class="stat-value-mobile">${probability}%</div>
                                </div>
                                <div class="stat-mobile">
                                    <span class="stat-label-mobile">Safety</span>
                                    <div class="stat-value-mobile">${Math.max(0, rankDiff)}</div>
                                </div>
                                <div class="stat-mobile">
                                    <span class="stat-label-mobile">Round</span>
                                    <div class="stat-value-mobile">${college.best_round?.split(' ')[0] || 'R1'}</div>
                                </div>
                            </div>
                            
                            <span class="safety-badge ${safetyClass}">${college.safety_level}</span>
                            <span class="round-info">${college.best_round || 'First Round'}</span>
                            
                            <button class="cutoff-toggle" onclick="toggleCutoffs('${college.college_code}')">
                                <i class="fas fa-eye"></i>View All Cutoffs
                            </button>
                            
                            <div id="cutoff-${college.college_code}" class="cutoff-details" style="display: none;">
                                <div class="cutoff-categories">
                                    <strong>All Category Cutoffs:</strong>
                                    <div id="categories-${college.college_code}">Loading...</div>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                html += '</div>';
                resultsDiv.innerHTML = html;
            } else {
                resultsDiv.innerHTML = `
                    <div class="no-results">
                        <h2><i class="fas fa-frown"></i>No colleges found</h2>
                        <p><strong>Rank ${data.student_rank}</strong> in <strong>${data.category}</strong> category</p>
                        <p><i class="fas fa-lightbulb"></i> Try other categories or check subsequent rounds</p>
                    </div>
                `;
            }
        }
        
        async function toggleCutoffs(collegeCode) {
            const detailsDiv = document.getElementById(`cutoff-${collegeCode}`);
            const categoriesDiv = document.getElementById(`categories-${collegeCode}`);
            
            if (detailsDiv.style.display === 'none') {
                detailsDiv.style.display = 'block';
                
                try {
                    const response = await fetch(`/api/college/${collegeCode}`);
                    const collegeData = await response.json();
                    
                    let cutoffHTML = '';
                    const cutoffs = collegeData.cutoffs || {};
                    
                    Object.entries(cutoffs).forEach(([category, cutoff]) => {
                        if (cutoff !== null) {
                            const isCurrentCategory = category === currentCategory;
                            const style = isCurrentCategory ? 'font-weight: bold; color: #667eea;' : '';
                            const indicator = isCurrentCategory ? ' ← You' : '';
                            
                            cutoffHTML += `
                                <div class="cutoff-row" style="${style}">
                                    <span>${category}${indicator}</span>
                                    <span>${cutoff}</span>
                                </div>
                            `;
                        }
                    });
                    
                    categoriesDiv.innerHTML = cutoffHTML || '<div>No cutoff data</div>';
                } catch (error) {
                    categoriesDiv.innerHTML = '<div>Error loading data</div>';
                }
            } else {
                detailsDiv.style.display = 'none';
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(MOBILE_HTML)

@app.route('/api/predict-mobile', methods=['POST'])
def predict_mobile():
    try:
        data = request.json
        student_rank = int(data['rank'])
        category = data['category']
        preferences = data.get('preferences', {})
        
        if predictor and predictor.is_trained:
            predictions = predictor.predict_with_intelligence(student_rank, category, preferences)
            eligible_colleges = [p for p in predictions if p['admission_probability'] > 0.2][:20]
        else:
            # Fallback basic search if ML model not available
            eligible_colleges = basic_search(student_rank, category, preferences)
        
        return jsonify({
            'success': True,
            'student_rank': student_rank,
            'category': category,
            'total_colleges': len(eligible_colleges),
            'colleges': eligible_colleges
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def basic_search(student_rank, category, preferences):
    """Fallback search when ML model is not available"""
    try:
        with open('combined_pgcet_data.json', 'r') as f:
            colleges = json.load(f)
        
        eligible = []
        for college in colleges:
            cutoffs = college.get('cutoffs', {})
            if category in cutoffs and cutoffs[category] is not None:
                cutoff = int(cutoffs[category]) if str(cutoffs[category]).isdigit() else None
                if cutoff and student_rank <= cutoff:
                    # City preference filter
                    if preferences.get('preferred_city'):
                        if preferences['preferred_city'].upper() not in college.get('location', '').upper():
                            continue
                    
                    eligible.append({
                        'college_code': college['collegeCode'],
                        'college_name': college['collegeName'],
                        'location': college['location'],
                        'city': college.get('city', college['location'].split(',')[-1].strip()),
                        'cutoff_rank': cutoff,
                        'best_round': 'First Round',
                        'admission_probability': 0.8,  # Default probability
                        'safety_level': 'Eligible',
                        'rank_difference': cutoff - student_rank,
                        'preference_match': False
                    })
        
        # Sort by cutoff rank
        eligible.sort(key=lambda x: x['cutoff_rank'])
        return eligible[:20]
    except:
        return []

@app.route('/api/college/<college_code>')
def get_college_details(college_code):
    try:
        with open('combined_pgcet_data.json', 'r') as f:
            colleges = json.load(f)
        
        college = next((c for c in colleges if c['collegeCode'] == college_code), None)
        if college:
            return jsonify(college)
        else:
            return jsonify({'error': 'College not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-status')
def get_data_status():
    try:
        # Get file modification time
        if os.path.exists('combined_pgcet_data.json'):
            mod_time = os.path.getmtime('combined_pgcet_data.json')
            last_updated = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
            
            # Count total colleges
            with open('combined_pgcet_data.json', 'r') as f:
                colleges = json.load(f)
                total_colleges = len(colleges)
        else:
            last_updated = 'File not found'
            total_colleges = 0
        
        return jsonify({
            'success': True,
            'last_updated': last_updated,
            'total_colleges': total_colleges,
            'model_status': 'Active' if predictor and predictor.is_trained else 'Fallback Mode'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/refresh-data', methods=['POST'])
def refresh_data():
    try:
        # In a real scenario, you would reload data from source
        # For now, we'll just reload the JSON file and model
        global predictor
        
        # Reload data
        if os.path.exists('combined_pgcet_data.json'):
            # Try to reload ML model
            try:
                from advanced_ml_predictor import AdvancedPGCETPredictor
                predictor = AdvancedPGCETPredictor.load_models('advanced_pgcet_model.pkl', 'combined_pgcet_data.json')
                print("✅ ML models reloaded successfully")
            except Exception as e:
                print(f"⚠️ Could not reload ML model: {e}")
            
            return jsonify({
                'success': True,
                'message': 'Data refreshed successfully',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Data file not found'
            }), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("📱 Mobile PGCET Cutoff Finder Starting...")
    print("🎯 Mobile-first design with FontAwesome icons")
    print("🔧 Toggle features: About & Refresh Data")
    print("🌐 Access: http://localhost:5000")
    app.run(debug=True, port=5000)
