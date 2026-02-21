# Production Deployment Guide: FoodSight AI 🚀

This guide explains how to deploy the **FoodSight AI** application to a production environment (Server, Cloud, or VPS) so that it is accessible via a public IP or Domain, rather than just `localhost`.

---

## 🏗️ Architecture Overview
In a production setup, we don't use the built-in Flask development server (`app.run()`). Instead, we use:
1.  **WSGI Server**: (Gunicorn for Linux, Waitress for Windows) to handle multiple requests efficiently.
2.  **Reverse Proxy**: (Nginx or Apache) to handle SSL (HTTPS), static files, and security.
3.  **Process Manager**: (Systemd or PM2) to ensure the app restarts automatically if the server reboots.

---

## 🌐 Option 1: Cloud Platform (Easiest)
Platforms like **Render**, **Railway**, or **Heroku** handle most of the scaling and SSL for you.

1.  **Connect GitHub**: Push your code to a GitHub repository.
2.  **Build Command**: `pip install -r requirements.txt`
3.  **Start Command**: `gunicorn FoodSight-AI-App.app:app` (Note: adjust path if necessary).
4.  **Environment Variables**: Add your `SECRET_KEY` and other `.env` variables in the platform dashboard.

---

## 🐧 Option 2: Linux Servers (Ubuntu/Debian) - Recommended
Standard for professional deployments.

### 1. Install System Dependencies
```bash
sudo apt update
sudo apt install python3-pip python3-venv nginx
```

### 2. Setup Virtual Environment
```bash
cd /path/to/Review_4/FoodSight-AI-App
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
```

### 3. Configure Gunicorn Service (Systemd)
Create a file: `/etc/systemd/system/foodsight.service`
```ini
[Unit]
Description=Gunicorn instance to serve FoodSight AI
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/Review_4/FoodSight-AI-App
Environment="PATH=/path/to/Review_4/FoodSight-AI-App/venv/bin"
ExecStart=/path/to/Review_4/FoodSight-AI-App/venv/bin/gunicorn --workers 3 --bind unix:foodsight.sock -m 007 app:app

[Install]
WantedBy=multi-user.target
```

### 4. Configure Nginx as Reverse Proxy
Create a file: `/etc/nginx/sites-available/foodsight`
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        include proxy_params;
        proxy_pass http://unix:/path/to/Review_4/FoodSight-AI-App/foodsight.sock;
    }
}
```

---

## 🪟 Option 3: Windows Production (Waitress)
If you must remain on Windows but want it accessible on the network.

1.  **Install Waitress**:
    ```bash
    pip install waitress
    ```
2.  **Create a Production Entry Point** (`prod_server.py`):
    ```python
    from waitress import serve
    from app import app
    
    if __name__ == '__main__':
        print("Serving FoodSight AI on port 5000...")
        serve(app, host='0:0:0:0', port=5000)
    ```
3.  **Open Firewall**: Ensure Port 5000 is open in Windows Firewall for public traffic.

---

## 🔒 Security Best Practices
- **Secret Key**: Never use a default secret key in production. Generate one: `python -c "import os; print(os.urandom(24).hex())"`
- **SSL**: Use **Certbot (Let's Encrypt)** for free HTTPS certificates: `sudo apt install certbot python3-certbot-nginx`
- **Debug Mode**: Ensure `app.debug = False` (default in most production launchers).
- **Environment Variables**: Use `.env` files and never commit them to public GitHub repos.

---

## 📁 Critical Directories
Ensure the following directories have **Write Permissions** for the server user:
- `instance/`: For the SQLite database (`users.db`).
- `uploads/`: For temporary image processing.

> [!TIP]
> **Performance Note**: TensorFlow can be memory intensive. Ensure your server has at least **2GB of RAM** (4GB recommended) to avoid `Out Of Memory` errors during image analysis.
