#!/bin/bash
# start_docker_services.sh - Start AIMS Docker services with proper error handling

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting AIMS Docker Services${NC}"
echo "================================"

# Check if docker is accessible
if ! docker ps >/dev/null 2>&1; then
    echo -e "${RED}❌ Cannot access Docker!${NC}"
    echo ""
    echo "This is likely a permissions issue. To fix it:"
    echo ""
    echo "1. Add yourself to the docker group:"
    echo -e "   ${BLUE}sudo usermod -aG docker $USER${NC}"
    echo ""
    echo "2. Apply the new group (choose one):"
    echo -e "   ${BLUE}newgrp docker${NC}  (applies immediately to this terminal)"
    echo "   OR"
    echo "   Log out and log back in (applies to all terminals)"
    echo ""
    echo "3. Then run this script again"
    exit 1
fi

echo -e "${GREEN}✅ Docker is accessible${NC}"

# Check if docker-compose exists
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ docker-compose not found${NC}"
    echo "Install it with:"
    echo -e "   ${BLUE}sudo apt install docker-compose${NC}"
    exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ docker-compose.yml not found${NC}"
    echo "Make sure you're in the AIMS project directory"
    exit 1
fi

# Start services
echo ""
echo -e "${BLUE}Starting services...${NC}"
docker-compose up -d

# Wait a moment for services to start
echo ""
echo -e "${BLUE}Waiting for services to initialize...${NC}"
sleep 5

# Check service status
echo ""
echo -e "${BLUE}Service Status:${NC}"
docker-compose ps

# Check if services are healthy
echo ""
echo -e "${BLUE}Checking service health...${NC}"

# Function to check if a service is running
check_service() {
    local service=$1
    local port=$2
    
    if docker-compose ps | grep -q "aims_${service}.*Up"; then
        echo -e "${GREEN}✅ ${service} is running${NC}"
        
        # Try to connect to the port
        if timeout 1 bash -c "echo >/dev/tcp/localhost/${port}" 2>/dev/null; then
            echo -e "   Port ${port} is accessible"
        else
            echo -e "   ${YELLOW}⚠️  Port ${port} may not be ready yet${NC}"
        fi
        return 0
    else
        echo -e "${RED}❌ ${service} is not running${NC}"
        return 1
    fi
}

# Check each service
all_good=true
check_service "postgres" 5433 || all_good=false
check_service "redis" 6379 || all_good=false
check_service "qdrant" 6333 || all_good=false

# Summary
echo ""
if $all_good; then
    echo -e "${GREEN}✅ All services are running!${NC}"
    echo ""
    echo "You can now:"
    echo "1. Activate the virtual environment:"
    echo -e "   ${BLUE}source venv/bin/activate${NC}"
    echo ""
    echo "2. Run AIMS:"
    echo -e "   ${BLUE}python -m src.main${NC}"
    echo ""
    echo "To stop services later:"
    echo -e "   ${BLUE}docker-compose down${NC}"
else
    echo -e "${YELLOW}⚠️  Some services failed to start${NC}"
    echo ""
    echo "Check the logs with:"
    echo -e "   ${BLUE}docker-compose logs${NC}"
    echo ""
    echo "Or check a specific service:"
    echo -e "   ${BLUE}docker-compose logs postgres${NC}"
    echo -e "   ${BLUE}docker-compose logs redis${NC}"
    echo -e "   ${BLUE}docker-compose logs qdrant${NC}"
fi