FROM python:3.11-slim-bookworm

# Optional: switch Debian APT mirror to improve connectivity (default: deb.debian.org)
# 可通过 --build-arg APT_MIRROR=https://mirrors.aliyun.com/debian 覆盖
ARG APT_MIRROR=https://mirrors.aliyun.com/debian
RUN echo "deb ${APT_MIRROR} bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list \
    && echo "deb ${APT_MIRROR}-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list \
    && echo "deb ${APT_MIRROR} bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list

# Set platform for multi-arch builds (Docker Buildx will set this)
ARG TARGETPLATFORM
ARG NODE_MAJOR=20

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    netcat-traditional \
    gnupg \
    curl \
    unzip \
    xvfb \
    libgconf-2-4 \
    libxss1 \
    libnss3 \
    libnspr4 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    xdg-utils \
    fonts-liberation \
    dbus \
    xauth \
    x11vnc \
    tigervnc-tools \
    supervisor \
    net-tools \
    procps \
    git \
    python3-numpy \
    fontconfig \
    fonts-dejavu \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install noVNC
RUN git clone https://github.com/novnc/noVNC.git /opt/novnc \
    && git clone https://github.com/novnc/websockify /opt/novnc/utils/websockify \
    && ln -s /opt/novnc/vnc.html /opt/novnc/index.html

# Install Node.js (Debian repo; bookworm ships Node 18.x)
RUN apt-get update \
    && apt-get install -y nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Verify Node.js and npm installation (optional, but good for debugging)
RUN node -v && npm -v && npx -v

# Set up working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Install playwright browsers and dependencies
# playwright documentation suggests PLAYWRIGHT_BROWSERS_PATH is still relevant
# or that playwright installs to a similar default location that Playwright would.
# Let's assume playwright respects PLAYWRIGHT_BROWSERS_PATH or its default install location is findable.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-browsers
RUN mkdir -p $PLAYWRIGHT_BROWSERS_PATH

# Install recommended: Google Chrome (instead of just Chromium for better undetectability)
# The 'playwright install chrome' command might download and place it.
# The '--with-deps' equivalent for playwright install is to run 'playwright install-deps chrome' after.
# RUN playwright install chrome --with-deps

# Alternative: Install Chromium if Google Chrome is problematic in certain environments
RUN playwright install chromium --with-deps


# Copy the application code
COPY . .

# Create tmp directory for http_run.py
RUN mkdir -p /app/tmp

# Expose ports: 7788 (webui), 6080 (novnc), 5901 (vnc), 9222 (debug), 9000 (http_run.py API)
EXPOSE 7788 6080 5901 9222 9000

# Use http_run.py instead of supervisord
CMD ["python", "http_run.py"]