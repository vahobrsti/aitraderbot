#!/bin/bash
# ============================================================
# Headless IB Gateway setup for IBIT options data (read-only)
# ============================================================
# Installs IB Gateway + IBC (IbcAlpha) and runs them as a systemd service
# under Xvfb (a virtual display), auto-logging in so the Django collector can
# read the IBIT option chain at 127.0.0.1:<port>.
#
# This path is READ-ONLY (ReadOnlyApi=yes) — it collects market data only and
# cannot place orders. That matches Phase 1 (publish signals to Telegram, no
# execution).
#
# Run AFTER setup-vps.sh, once you have IBKR credentials. Pass credentials via
# the environment (they are NOT stored in the repo or in Django's .env):
#
#   sudo -E IBKR_USERNAME=youruser IBKR_PASSWORD='yourpass' \
#           IBKR_TRADING_MODE=paper \
#           bash deploy/setup-ibkr-gateway.sh
#
# Security notes:
#   - Credentials are written only to /etc/ibkr-gateway.env (mode 600).
#   - The API socket binds to localhost; never open its port in iptables.
#   - Use a PAPER login for an unattended server (avoids daily 2FA prompts).
#
# Version notes (verify current values before running — IBKR/IBC change these):
#   - IB Gateway stable standalone installer URL (below)
#   - IBC release version (IBC_VERSION below)
#   - After install, TWS_MAJOR_VRSN is auto-detected from the install dir.
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ------------------------------------------------------------
# Configuration (override via environment)
# ------------------------------------------------------------
PROJECT_USER="${PROJECT_USER:-deploy}"
USER_HOME="$(getent passwd "${PROJECT_USER}" | cut -d: -f6)"
IBC_VERSION="${IBC_VERSION:-3.20.0}"          # verify at github.com/IbcAlpha/IBC/releases
IBC_PATH="/opt/ibc"
TWS_PATH="${USER_HOME}/Jts"                    # IB Gateway install dir
IBC_INI="${USER_HOME}/ibc/config.ini"
LOG_PATH="${USER_HOME}/ibc/logs"
ENV_FILE="/etc/ibkr-gateway.env"
GATEWAY_INSTALLER_URL="https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh"

IBKR_TRADING_MODE="${IBKR_TRADING_MODE:-paper}"   # paper | live
IBKR_PORT="${IBKR_PORT:-4002}"                    # 4002 paper, 4001 live
DISPLAY_NUM="${DISPLAY_NUM:-1}"

# ------------------------------------------------------------
# Preconditions
# ------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
    log_error "Run as root (use: sudo -E bash $0)"; exit 1
fi
if [[ -z "${IBKR_USERNAME:-}" || -z "${IBKR_PASSWORD:-}" ]]; then
    log_error "IBKR_USERNAME and IBKR_PASSWORD must be set in the environment."
    log_error "Example: sudo -E IBKR_USERNAME=u IBKR_PASSWORD=p IBKR_TRADING_MODE=paper bash $0"
    exit 1
fi
if [[ -z "${USER_HOME}" || ! -d "${USER_HOME}" ]]; then
    log_error "Home directory for user '${PROJECT_USER}' not found."; exit 1
fi
if [[ "${IBKR_TRADING_MODE}" == "live" ]]; then
    log_warn "TRADING_MODE=live. Read-only API is still enforced, but a live login"
    log_warn "is subject to daily 2FA. A paper login is strongly recommended here."
fi

log_info "Installing dependencies (xvfb, xterm, JRE, unzip)..."
apt-get update -qq
# xterm is required: IBC's gatewaystart.sh launches the Gateway inside an xterm
# (it runs fine under the Xvfb virtual display). Without it the service
# crash-loops with "xterm: command not found".
apt-get install -y xvfb xterm default-jre unzip curl >/dev/null

# ------------------------------------------------------------
# 1. Install IB Gateway (unattended)
# ------------------------------------------------------------
log_info "Downloading IB Gateway installer..."
tmp_installer="/tmp/ibgateway-standalone.sh"
curl -fsSL "${GATEWAY_INSTALLER_URL}" -o "${tmp_installer}"
chmod +x "${tmp_installer}"

log_info "Installing IB Gateway to ${TWS_PATH} (quiet mode)..."
# install4j: -q unattended, -dir target. Runs as the project user.
sudo -u "${PROJECT_USER}" bash -c "yes '' | '${tmp_installer}' -q -dir '${TWS_PATH}'" || {
    log_warn "Quiet install returned non-zero; verify ${TWS_PATH}/ibgateway exists."
}
rm -f "${tmp_installer}"

# Detect installed major version (folder name, e.g. '10.30').
if [[ -d "${TWS_PATH}/ibgateway" ]]; then
    TWS_MAJOR_VRSN="$(ls -1 "${TWS_PATH}/ibgateway" | sort -V | tail -1)"
else
    log_warn "Could not find ${TWS_PATH}/ibgateway; defaulting TWS_MAJOR_VRSN=10.30 (verify)."
    TWS_MAJOR_VRSN="10.30"
fi
log_info "Detected IB Gateway version: ${TWS_MAJOR_VRSN}"

# ------------------------------------------------------------
# 2. Install IBC
# ------------------------------------------------------------
log_info "Installing IBC ${IBC_VERSION} to ${IBC_PATH}..."
mkdir -p "${IBC_PATH}"
tmp_ibc="/tmp/IBCLinux-${IBC_VERSION}.zip"
curl -fsSL "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip" -o "${tmp_ibc}"
unzip -o -q "${tmp_ibc}" -d "${IBC_PATH}"
chmod +x "${IBC_PATH}"/*.sh "${IBC_PATH}/scripts/"*.sh 2>/dev/null || true
rm -f "${tmp_ibc}"

# ------------------------------------------------------------
# 3. IBC config.ini (read-only API, auto-restart to avoid daily logoff)
# ------------------------------------------------------------
log_info "Writing IBC config to ${IBC_INI}..."
mkdir -p "$(dirname "${IBC_INI}")" "${LOG_PATH}"
cat > "${IBC_INI}" <<EOF
# IBC configuration — generated by setup-ibkr-gateway.sh
# Credentials are supplied via the environment (see ${ENV_FILE}), not stored here.
LoginId=
Password=
TradingMode=${IBKR_TRADING_MODE}

# --- API access (read-only: data collection only, no order placement) ---
ReadOnlyApi=yes
AcceptIncomingConnectionAction=accept
AllowBlindTrading=no
OverrideTwsApiPort=${IBKR_PORT}

# --- Session behaviour ---
# Auto-restart (rather than auto-logoff) keeps the session alive across IBKR's
# forced daily reset without requiring a fresh login/2FA.
IbAutoClosedown=no
ClosedownAt=
AutoRestartTime=11:45 PM
ExistingSessionDetectedAction=primary
DismissPasswordExpiryWarning=yes
DismissNSEComplianceNotice=yes
EOF
chown -R "${PROJECT_USER}:${PROJECT_USER}" "$(dirname "${IBC_INI}")"
chmod 600 "${IBC_INI}"

# ------------------------------------------------------------
# 4. Protected environment file (credentials live here only)
# ------------------------------------------------------------
log_info "Writing ${ENV_FILE} (mode 600)..."
cat > "${ENV_FILE}" <<EOF
TWSUSERID=${IBKR_USERNAME}
TWSPASSWORD=${IBKR_PASSWORD}
TRADING_MODE=${IBKR_TRADING_MODE}
TWS_MAJOR_VRSN=${TWS_MAJOR_VRSN}
IBC_INI=${IBC_INI}
IBC_PATH=${IBC_PATH}
TWS_PATH=${TWS_PATH}
TWS_SETTINGS_PATH=${TWS_PATH}
LOG_PATH=${LOG_PATH}
EOF
chown root:"${PROJECT_USER}" "${ENV_FILE}"
chmod 640 "${ENV_FILE}"

# ------------------------------------------------------------
# 5. systemd service (headless via Xvfb)
# ------------------------------------------------------------
log_info "Creating systemd service ibkr-gateway.service..."
cat > /etc/systemd/system/ibkr-gateway.service <<EOF
[Unit]
Description=IB Gateway (IBC, headless) for IBIT options data
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${PROJECT_USER}
Group=${PROJECT_USER}
EnvironmentFile=${ENV_FILE}
Environment=DISPLAY=:${DISPLAY_NUM}
# Xvfb provides a virtual display so the Gateway GUI can run with no monitor.
ExecStart=/usr/bin/xvfb-run -a -n ${DISPLAY_NUM} -s "-screen 0 1024x768x24" ${IBC_PATH}/gatewaystart.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ibkr-gateway.service
systemctl restart ibkr-gateway.service

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
echo ""
echo "============================================================"
log_info "IB Gateway (headless) setup complete"
echo "============================================================"
echo "  Trading mode : ${IBKR_TRADING_MODE}"
echo "  API port     : ${IBKR_PORT} (localhost only, READ-ONLY)"
echo "  Gateway ver  : ${TWS_MAJOR_VRSN}"
echo "  IBC config   : ${IBC_INI}"
echo "  Credentials  : ${ENV_FILE} (mode 640)"
echo "  Logs         : ${LOG_PATH}"
echo ""
echo "  Check status : sudo systemctl status ibkr-gateway"
echo "  Follow logs  : sudo journalctl -u ibkr-gateway -f"
echo ""
echo "  Verify data (during US market hours, 14:00-20:00 UTC weekdays):"
echo "     cd ${USER_HOME%/*}/app 2>/dev/null || cd /var/www/app"
echo "     source venv/bin/activate"
echo "     python manage.py collect_options --exchange ibkr --dry-run"
echo ""
log_warn "First login may require a one-time 2FA approval on IBKR Mobile."
log_warn "Confirm IBKR_PORT here matches IBKR_PORT in .env.production."
echo "============================================================"
