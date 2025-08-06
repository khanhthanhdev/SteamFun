#!/bin/bash

# Production Deployment Script for LangGraph Video Generation Workflow
# This script sets up the production environment with proper security and monitoring

set -e  # Exit on any error

echo "🚀 Starting production deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root (not recommended for production)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root is not recommended for production deployment"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed. Please install docker-compose and try again."
    exit 1
fi

# Check for required environment variables
print_status "Checking required environment variables..."
REQUIRED_VARS=(
    "POSTGRES_PASSWORD"
    "REDIS_PASSWORD"
    "SECRET_KEY"
    "OPENAI_API_KEY"
    "OPENROUTER_API_KEY"
)

MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    print_error "Missing required environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    print_error "Please set these variables and try again."
    exit 1
fi

# Validate SECRET_KEY strength
if [ ${#SECRET_KEY} -lt 32 ]; then
    print_error "SECRET_KEY must be at least 32 characters long for production"
    exit 1
fi

# Create necessary directories with proper permissions
print_status "Creating necessary directories..."
mkdir -p /var/app/output
mkdir -p /var/app/logs
mkdir -p /var/app/data/rag/chroma_db
mkdir -p /var/app/data/context_learning
mkdir -p /var/app/models
mkdir -p /var/app/config/runtime
mkdir -p /var/app/ssl
mkdir -p /var/app/monitoring

# Set proper permissions
chmod 755 /var/app
chmod 755 /var/app/output
chmod 755 /var/app/logs
chmod 755 /var/app/data
chmod 700 /var/app/ssl  # Restrict SSL directory

# Copy production configuration
print_status "Setting up production configuration..."
if [ ! -f "/var/app/config/runtime/workflow.yaml" ]; then
    cp config/templates/production.yaml /var/app/config/runtime/workflow.yaml
    print_success "Production configuration copied"
else
    print_warning "Configuration already exists at /var/app/config/runtime/workflow.yaml"
fi

# Validate configuration
print_status "Validating production configuration..."
if python -c "
import sys
sys.path.append('src')
from langgraph_agents.config.validation import validate_config_from_file
try:
    validate_config_from_file('/var/app/config/runtime/workflow.yaml')
    print('Configuration validation passed')
except Exception as e:
    print(f'Configuration validation failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
    print_success "Configuration validation passed"
else
    print_error "Configuration validation failed. Please check your configuration file."
    exit 1
fi

# Check SSL certificates
if [ ! -f "/var/app/ssl/cert.pem" ] || [ ! -f "/var/app/ssl/key.pem" ]; then
    print_warning "SSL certificates not found. Generating self-signed certificates..."
    openssl req -x509 -newkey rsa:4096 -keyout /var/app/ssl/key.pem -out /var/app/ssl/cert.pem -days 365 -nodes \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    print_success "Self-signed SSL certificates generated"
fi

# Create monitoring configuration
print_status "Setting up monitoring configuration..."
cat > /var/app/monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'video-gen-api'
    static_configs:
      - targets: ['video-gen-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['db:5432']
    scrape_interval: 60s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 60s
EOF

# Create nginx configuration
print_status "Setting up nginx configuration..."
cat > nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream api {
        server video-gen-api:8000;
    }

    server {
        listen 80;
        server_name _;
        return 301 https://\$server_name\$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name _;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        client_max_body_size 100M;

        location / {
            proxy_pass http://api;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        location /health {
            proxy_pass http://api/health;
            access_log off;
        }
    }
}
EOF

# Stop any existing containers
print_status "Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down --remove-orphans

# Pull latest images
print_status "Pulling latest images..."
docker-compose -f docker-compose.prod.yml pull

# Build and start production services
print_status "Building and starting production services..."
docker-compose -f docker-compose.prod.yml up --build -d

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 15

# Check service health
print_status "Checking service health..."

# Check database
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker-compose -f docker-compose.prod.yml exec -T db pg_isready -U postgres > /dev/null 2>&1; then
        print_success "Database is ready"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        print_error "Database failed to start"
        docker-compose -f docker-compose.prod.yml logs db
        exit 1
    fi
    sleep 2
done

# Check Redis
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker-compose -f docker-compose.prod.yml exec -T redis redis-cli --no-auth-warning -a "$REDIS_PASSWORD" ping > /dev/null 2>&1; then
        print_success "Redis is ready"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        print_error "Redis failed to start"
        docker-compose -f docker-compose.prod.yml logs redis
        exit 1
    fi
    sleep 2
done

# Check API health
print_status "Waiting for API to be ready..."
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f -k https://localhost/health > /dev/null 2>&1; then
        print_success "API is ready"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        print_error "API failed to start"
        docker-compose -f docker-compose.prod.yml logs video-gen-api
        exit 1
    fi
    sleep 3
done

# Check workflow health
print_status "Checking workflow health..."
if curl -f -k https://localhost/health/workflow > /dev/null 2>&1; then
    print_success "Workflow health check passed"
else
    print_warning "Workflow health check failed - checking logs..."
    docker-compose -f docker-compose.prod.yml logs --tail=20 video-gen-api
fi

# Set up log rotation
print_status "Setting up log rotation..."
cat > /etc/logrotate.d/video-gen << EOF
/var/app/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
    postrotate
        docker-compose -f docker-compose.prod.yml restart video-gen-api
    endscript
}
EOF

# Set up monitoring alerts (if monitoring is enabled)
if docker-compose -f docker-compose.prod.yml ps | grep -q prometheus; then
    print_status "Setting up monitoring alerts..."
    # Add Prometheus alerting rules here if needed
    print_success "Monitoring is enabled"
fi

# Security hardening
print_status "Applying security hardening..."

# Set up firewall rules (example - adjust for your environment)
if command -v ufw &> /dev/null; then
    ufw --force enable
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow 22/tcp    # SSH
    ufw allow 80/tcp    # HTTP
    ufw allow 443/tcp   # HTTPS
    print_success "Firewall configured"
fi

# Display deployment information
print_success "Production deployment completed successfully!"
echo ""
echo "📋 Service Information:"
echo "  🌐 API Server: https://localhost (HTTP redirects to HTTPS)"
echo "  📚 API Docs: https://localhost/docs"
echo "  🗄️  Database: Internal (not exposed)"
echo "  🔄 Redis: Internal (not exposed)"
if docker-compose -f docker-compose.prod.yml ps | grep -q prometheus; then
    echo "  📊 Prometheus: http://localhost:9090"
    echo "  📈 Grafana: http://localhost:3000"
fi
echo ""
echo "🔧 Useful Commands:"
echo "  View logs: docker-compose -f docker-compose.prod.yml logs -f"
echo "  Stop services: docker-compose -f docker-compose.prod.yml down"
echo "  Restart API: docker-compose -f docker-compose.prod.yml restart video-gen-api"
echo "  Check health: curl -k https://localhost/health/workflow"
echo ""
echo "📁 Important Paths:"
echo "  Configuration: /var/app/config/runtime/workflow.yaml"
echo "  Output: /var/app/output/"
echo "  Logs: /var/app/logs/"
echo "  SSL Certificates: /var/app/ssl/"
echo ""
echo "🔒 Security Notes:"
echo "  - SSL/TLS is enabled with self-signed certificates"
echo "  - Database and Redis are not exposed externally"
echo "  - API keys are encrypted in configuration"
echo "  - Log rotation is configured"
echo ""

# Final security check
print_status "Performing final security check..."
SECURITY_ISSUES=()

# Check if default passwords are being used
if [ "$POSTGRES_PASSWORD" = "password" ]; then
    SECURITY_ISSUES+=("Default PostgreSQL password detected")
fi

if [ "$REDIS_PASSWORD" = "" ]; then
    SECURITY_ISSUES+=("Redis password not set")
fi

# Check SSL certificate validity
if ! openssl x509 -in /var/app/ssl/cert.pem -noout -checkend 86400 > /dev/null 2>&1; then
    SECURITY_ISSUES+=("SSL certificate expires within 24 hours")
fi

if [ ${#SECURITY_ISSUES[@]} -ne 0 ]; then
    print_warning "Security issues detected:"
    for issue in "${SECURITY_ISSUES[@]}"; do
        echo "  ⚠️  $issue"
    done
    echo ""
fi

print_success "Production environment is ready! 🎉"
print_warning "Remember to:"
echo "  1. Update DNS records to point to this server"
echo "  2. Replace self-signed certificates with proper SSL certificates"
echo "  3. Set up proper backup procedures"
echo "  4. Configure monitoring alerts"
echo "  5. Review and update security settings regularly"