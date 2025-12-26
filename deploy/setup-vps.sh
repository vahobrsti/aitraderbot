#!/bin/bash
# ============================================================
# VPS Setup Script for Django + Caddy on Debian 12
# Domain: options.somimobile.com
# ============================================================

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================
# CONFIGURATION - Server Settings
# ============================================================
DOMAIN="options.somimobile.com"
PROJECT_NAME="aitrader"
PROJECT_USER="deploy"
PROJECT_DIR="/var/www/app"
VENV_DIR="${PROJECT_DIR}/venv"
REPO_URL="${REPO_URL:-https://github.com/vahobrsti/aitraderbot.git}"

# ============================================================
# DATABASE CONFIGURATION (Override via environment or prompted)
# ============================================================
DB_NAME="${DB_NAME:-aitrader_db}"
DB_USER="${DB_USER:-aitrader_user}"
DB_PASSWORD="${DB_PASSWORD:-$(openssl rand -base64 32)}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

# ============================================================
# DJANGO CONFIGURATION (Override via environment)
# ============================================================
DJANGO_SECRET_KEY="${DJANGO_SECRET_KEY:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(50))')}"
DJANGO_DEBUG="${DJANGO_DEBUG:-False}"
ALLOWED_HOSTS="${ALLOWED_HOSTS:-${DOMAIN}}"

# ============================================================
# GOOGLE SHEETS CONFIGURATION (Override via environment)
# ============================================================
GSPREAD_SHEET_ID="${GSPREAD_SHEET_ID:-}"
GSPREAD_CREDS_FILE="${GSPREAD_CREDS_FILE:-${PROJECT_DIR}/credentials/gspread-creds.json}"
GSPREAD_CREDS_JSON="${GSPREAD_CREDS_JSON:-}"

# ============================================================
# API CONFIGURATION (Override via environment)
# ============================================================
API_TOKEN="${API_TOKEN:-$(openssl rand -hex 32)}"

# ============================================================
# STEP 1: System Update & Base Packages
# ============================================================
log_info "Step 1: Updating system and installing base packages..."

apt update && apt upgrade -y
apt install -y \
    python3 python3-pip python3-venv python3-dev \
    git curl wget \
    build-essential libpq-dev \
    supervisor \
    iptables iptables-persistent \
    debian-keyring debian-archive-keyring apt-transport-https

# ============================================================
# STEP 2: Create Deploy User
# ============================================================
log_info "Step 2: Creating deploy user..."

if id "${PROJECT_USER}" &>/dev/null; then
    log_warn "User ${PROJECT_USER} already exists, skipping..."
else
    useradd -m -s /bin/bash ${PROJECT_USER}
    usermod -aG sudo ${PROJECT_USER}
    log_info "User ${PROJECT_USER} created"
fi

# ============================================================
# STEP 3: Configure Firewall (iptables)
# ============================================================
log_info "Step 3: Configuring iptables firewall..."

# Flush existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

# Set default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (port 22)
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP (port 80)
iptables -A INPUT -p tcp --dport 80 -j ACCEPT

# Allow HTTPS (port 443)
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow ping (optional, comment out if not desired)
iptables -A INPUT -p icmp --icmp-type echo-request -j ACCEPT

# Log dropped packets (optional)
iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "iptables-dropped: " --log-level 4

# Save rules to persist across reboots
netfilter-persistent save

log_info "iptables configured and saved"

# ============================================================
# STEP 4: Install PostgreSQL
# ============================================================
log_info "Step 4: Installing and configuring PostgreSQL..."

apt install -y postgresql postgresql-contrib

# Start PostgreSQL
systemctl start postgresql
systemctl enable postgresql

# Create database and user
sudo -u postgres psql <<EOF
CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';
CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};
ALTER USER ${DB_USER} CREATEDB;
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};
EOF

log_info "PostgreSQL configured. Database: ${DB_NAME}, User: ${DB_USER}"

# ============================================================
# STEP 5: Install Caddy
# ============================================================
log_info "Step 5: Installing Caddy..."

curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
apt update
apt install -y caddy

systemctl enable caddy

# ============================================================
# STEP 6: Clone Project & Setup Virtual Environment
# ============================================================
log_info "Step 6: Setting up project..."

mkdir -p ${PROJECT_DIR}
chown ${PROJECT_USER}:${PROJECT_USER} ${PROJECT_DIR}

# NOTE: Clone manually or uncomment below after setting REPO_URL
sudo -u ${PROJECT_USER} git clone ${REPO_URL} ${PROJECT_DIR}

# Create virtual environment
sudo -u ${PROJECT_USER} python3 -m venv ${VENV_DIR}

# Create directories
mkdir -p ${PROJECT_DIR}/staticfiles
mkdir -p ${PROJECT_DIR}/media
mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${PROJECT_DIR}/credentials
chown -R ${PROJECT_USER}:${PROJECT_USER} ${PROJECT_DIR}

# ============================================================
# STEP 7: Create Environment File
# ============================================================
log_info "Step 7: Creating environment file..."

cat > ${PROJECT_DIR}/.env.production <<EOF
# ===========================================
# Django Core Settings
# ===========================================
DEBUG=${DJANGO_DEBUG}
SECRET_KEY=${DJANGO_SECRET_KEY}
ALLOWED_HOSTS=${ALLOWED_HOSTS}

# ===========================================
# Database Configuration
# ===========================================
DB_NAME=${DB_NAME}
DB_USER=${DB_USER}
DB_PASSWORD=${DB_PASSWORD}
DB_HOST=${DB_HOST}
DB_PORT=${DB_PORT}

# ===========================================
# Security Settings
# ===========================================
CSRF_TRUSTED_ORIGINS=https://${DOMAIN}

# ===========================================
# Static & Media Files
# ===========================================
STATIC_ROOT=${PROJECT_DIR}/staticfiles
MEDIA_ROOT=${PROJECT_DIR}/media

# ===========================================
# Google Sheets Integration
# ===========================================
GSPREAD_SHEET_ID=${GSPREAD_SHEET_ID}
GSPREAD_CREDS_FILE=${GSPREAD_CREDS_FILE}
GSPREAD_CREDS_JSON=${GSPREAD_CREDS_JSON}

# ===========================================
# API Authentication
# ===========================================
API_TOKEN=${API_TOKEN}
EOF

chown ${PROJECT_USER}:${PROJECT_USER} ${PROJECT_DIR}/.env.production
chmod 600 ${PROJECT_DIR}/.env.production

# Create symlink so Django's load_dotenv('.env') works
ln -sf ${PROJECT_DIR}/.env.production ${PROJECT_DIR}/.env
chown -h ${PROJECT_USER}:${PROJECT_USER} ${PROJECT_DIR}/.env

log_info "Environment file created at ${PROJECT_DIR}/.env.production"
log_info "Symlink created: .env -> .env.production"

# ============================================================
# STEP 8: Create Gunicorn Systemd Service
# ============================================================
log_info "Step 8: Creating Gunicorn service..."

cat > /etc/systemd/system/gunicorn.service <<EOF
[Unit]
Description=Gunicorn daemon for ${PROJECT_NAME}
Requires=gunicorn.socket
After=network.target

[Service]
Type=notify
User=${PROJECT_USER}
Group=${PROJECT_USER}
WorkingDirectory=${PROJECT_DIR}
Environment="PATH=${VENV_DIR}/bin"
EnvironmentFile=${PROJECT_DIR}/.env.production
ExecStart=${VENV_DIR}/bin/gunicorn \\
    --access-logfile ${PROJECT_DIR}/logs/gunicorn-access.log \\
    --error-logfile ${PROJECT_DIR}/logs/gunicorn-error.log \\
    --workers 3 \\
    --bind unix:/run/gunicorn.sock \\
    ${PROJECT_NAME}.wsgi:application
ExecReload=/bin/kill -s HUP \$MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/gunicorn.socket <<EOF
[Unit]
Description=Gunicorn socket for ${PROJECT_NAME}

[Socket]
ListenStream=/run/gunicorn.sock
SocketUser=www-data

[Install]
WantedBy=sockets.target
EOF

systemctl daemon-reload
systemctl enable gunicorn.socket

# ============================================================
# STEP 9: Configure Caddy
# ============================================================
log_info "Step 9: Configuring Caddy..."

cat > /etc/caddy/Caddyfile <<EOF
${DOMAIN} {
    # Enable compression
    encode gzip

    # Serve static files directly
    handle_path /static/* {
        root * ${PROJECT_DIR}/staticfiles
        file_server
    }

    # Serve media files directly
    handle_path /media/* {
        root * ${PROJECT_DIR}/media
        file_server
    }

    # Proxy everything else to Gunicorn
    reverse_proxy unix//run/gunicorn.sock {
        header_up X-Forwarded-Proto {scheme}
        header_up X-Real-IP {remote_host}
    }

    # Security headers
    header {
        X-Content-Type-Options nosniff
        X-Frame-Options DENY
        Referrer-Policy strict-origin-when-cross-origin
    }

    # Logging
    log {
        output file /var/log/caddy/${DOMAIN}.log
    }
}
EOF

mkdir -p /var/log/caddy
caddy fmt --overwrite /etc/caddy/Caddyfile
systemctl restart caddy

# ============================================================
# STEP 10: Create Deploy Script
# ============================================================
log_info "Step 10: Creating deployment helper script..."

cat > ${PROJECT_DIR}/deploy.sh <<'DEPLOY_SCRIPT'
#!/bin/bash
set -e

cd /var/www/app
source venv/bin/activate
set -a; source .env.production; set +a

echo "Pulling latest code..."
git pull origin main

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Restarting Gunicorn..."
sudo systemctl restart gunicorn

echo "Deployment complete!"
DEPLOY_SCRIPT

chmod +x ${PROJECT_DIR}/deploy.sh
chown ${PROJECT_USER}:${PROJECT_USER} ${PROJECT_DIR}/deploy.sh

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "============================================================"
log_info "VPS Setup Complete!"
echo "============================================================"
echo ""
echo "Domain: https://${DOMAIN}"
echo "Project Directory: ${PROJECT_DIR}"
echo "Virtual Environment: ${VENV_DIR}"
echo ""
echo "============================================================"
echo "CREDENTIALS - SAVE THESE SECURELY!"
echo "============================================================"
echo ""
echo "Database:"
echo "  DB_NAME=${DB_NAME}"
echo "  DB_USER=${DB_USER}"
echo "  DB_PASSWORD=${DB_PASSWORD}"
echo "  DB_HOST=${DB_HOST}"
echo "  DB_PORT=${DB_PORT}"
echo ""
echo "Django:"
echo "  SECRET_KEY=${DJANGO_SECRET_KEY}"
echo ""
echo "API:"
echo "  API_TOKEN=${API_TOKEN}"
echo ""
echo "============================================================"
echo "NEXT STEPS"
echo "============================================================"
echo ""
echo "  1. Clone your repo to ${PROJECT_DIR}"
echo "  2. Update values in ${PROJECT_DIR}/.env.production"
echo "     - Add GSPREAD_SHEET_ID"
echo "     - Copy gspread credentials to ${GSPREAD_CREDS_FILE}"
echo "  3. Run: source ${VENV_DIR}/bin/activate"
echo "  4. Run: pip install -r requirements.txt"
echo "  5. Run: python manage.py migrate"
echo "  6. Run: python manage.py collectstatic"
echo "  7. Run: sudo systemctl start gunicorn.socket"
echo "  8. Point your DNS A record to this server's IP"
echo ""
echo "============================================================"
echo "USEFUL COMMANDS"
echo "============================================================"
echo ""
echo "  Check Gunicorn:    sudo systemctl status gunicorn"
echo "  Check Caddy:       sudo systemctl status caddy"
echo "  View logs:         sudo journalctl -u gunicorn -f"
echo "  Redeploy:          ${PROJECT_DIR}/deploy.sh"
echo "  iptables status:   sudo iptables -L -n -v"
echo ""
