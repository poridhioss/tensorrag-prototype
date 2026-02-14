#!/bin/bash

# Quick Test Script for TensorRag
# This script helps verify the setup is working

set -e

echo "🧪 TensorRag Quick Test"
echo "========================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Backend Health
echo "1. Testing Backend Health..."
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo -e "${GREEN}✓${NC} Backend is running"
else
    echo -e "${RED}✗${NC} Backend is not running. Start it with:"
    echo "   cd backend && source .venv/bin/activate && uvicorn app.main:app --reload"
    exit 1
fi

# Test 2: Cards Endpoint
echo ""
echo "2. Testing Cards Endpoint..."
CARDS=$(curl -s http://localhost:8000/api/cards)
CARD_COUNT=$(echo $CARDS | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")

if [ "$CARD_COUNT" -gt "0" ]; then
    echo -e "${GREEN}✓${NC} Cards endpoint working ($CARD_COUNT cards available)"
else
    echo -e "${RED}✗${NC} Cards endpoint not working"
    exit 1
fi

# Test 3: Modal Deployment
echo ""
echo "3. Testing Modal Deployment..."
cd backend
source .venv/bin/activate 2>/dev/null || true

if modal app list 2>/dev/null | grep -q "tensorrag"; then
    echo -e "${GREEN}✓${NC} Modal app 'tensorrag' is deployed"
else
    echo -e "${YELLOW}⚠${NC} Modal app not found. Deploy with:"
    echo "   modal deploy cards/modal_app.py"
fi

# Test 4: Frontend
echo ""
echo "4. Testing Frontend..."
if curl -s http://localhost:3000 > /dev/null; then
    echo -e "${GREEN}✓${NC} Frontend is running"
else
    echo -e "${YELLOW}⚠${NC} Frontend is not running. Start it with:"
    echo "   cd frontend && npm run dev"
fi

# Test 5: Storage Directory
echo ""
echo "5. Checking Storage Directory..."
if [ -d "backend/storage" ]; then
    echo -e "${GREEN}✓${NC} Storage directory exists"
else
    echo -e "${YELLOW}⚠${NC} Storage directory will be created automatically"
fi

echo ""
echo "========================"
echo -e "${GREEN}✅ Basic tests complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Open http://localhost:3000 in your browser"
echo "2. Try creating a simple pipeline:"
echo "   Data Load → Data Split → Model Define → Train → Evaluate"
echo "3. Configure and run the pipeline"
echo ""
