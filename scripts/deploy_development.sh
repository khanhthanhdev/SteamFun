#!/bin/bash

# Development Deployment Script for LangGraph Video Generation Workflow
# This script sets up the development environment with proper configuration

set -e  # Exit on any error

echo "🚀 Starting development deployment..."

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

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p output/dev
mkdir -p logs
mkdir -p data/rag/chroma_db
mkdir -p data/context_learning
mkdir -p models
mkdir -p config/runtime

# Copy development configuration if it doesn't exist
if [ ! -f "config/runtime/workflow.yaml" ]; then
    print_status "Copying development configuration..."
    cp config/templates/development.yaml config/runtime/workflow.yaml
    print_success "Development configuration copied"
else
    print_warning "Configuration already exists at config/runtime/workflow.yaml"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Copying from .env.example..."
    cp .env.example .env
    print_warning "Please edit .env file with your API keys and configuration"
else
    print_status ".env file found"
fi

# Validate configuration
print_status "Validating configuration..."
if python -c "
import sys
sys.path.append('src')
from langgraph_agents.config.validation import validate_config_from_file
try:
    validate_config_from_file('config/runtime/workflow.yaml')
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

# Stop any existing containers
print_status "Stopping existing containers..."
docker-compose down --remove-orphans

# Build and start services
print_status "Building and starting development services..."
docker-compose up --build -d

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 10

# Check service health
print_status "Checking service health..."

# Check database
if docker-compose exec -T db pg_isready -U postgres > /dev/null 2>&1; then
    print_success "Database is ready"
else
    print_error "Database is not ready"
    exit 1
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    print_success "Redis is ready"
else
    print_error "Redis is not ready"
    exit 1
fi

# Check API health
print_status "Waiting for API to be ready..."
for i in {1..30}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "API failed to start within 30 attempts"
        docker-compose logs video-gen-api
        exit 1
    fi
    sleep 2
done

# Check workflow health
print_status "Checking workflow health..."
if curl -f http://localhost:8000/health/workflow > /dev/null 2>&1; then
    print_success "Workflow health check passed"
else
    print_warning "Workflow health check failed - this may be normal on first startup"
fi

# Display service information
print_success "Development deployment completed successfully!"
echo ""
echo "📋 Service Information:"
echo "  🌐 API Server: http://localhost:8000"
echo "  📚 API Docs: http://localhost:8000/docs"
echo "  🎛️  Gradio UI: http://localhost:7860 (if enabled)"
echo "  🗄️  Database: localhost:5432"
echo "  🔄 Redis: localhost:6379"
echo ""
echo "🔧 Useful Commands:"
echo "  View logs: docker-compose logs -f"
echo "  Stop services: docker-compose down"
echo "  Restart API: docker-compose restart video-gen-api"
echo "  Access database: docker-compose exec db psql -U postgres -d videogen"
echo "  Access Redis: docker-compose exec redis redis-cli"
echo ""
echo "📁 Important Paths:"
echo "  Configuration: config/runtime/workflow.yaml"
echo "  Output: output/dev/"
echo "  Logs: logs/"
echo ""

# Check for common issues
print_status "Checking for common issues..."

# Check disk space
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    print_warning "Disk usage is high (${DISK_USAGE}%). Consider cleaning up to avoid issues."
fi

# Check memory
if command -v free &> /dev/null; then
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ "$MEMORY_USAGE" -gt 80 ]; then
        print_warning "Memory usage is high (${MEMORY_USAGE}%). Consider closing other applications."
    fi
fi

# Check if API keys are set
if grep -q "your_.*_api_key_here" .env 2>/dev/null; then
    print_warning "Some API keys appear to be using default values. Please update your .env file."
fi

print_success "Development environment is ready! 🎉"