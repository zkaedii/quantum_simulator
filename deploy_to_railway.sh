#!/bin/bash

# 🚀 QUANTUM PLATFORM - RAILWAY DEPLOYMENT SCRIPT
# ===============================================
# Automated deployment of your quantum computing platform to Railway

echo "🌟 DEPLOYING QUANTUM PLATFORM TO RAILWAY 🌟"
echo "============================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please log in to Railway..."
    railway login
fi

# Initialize Railway project if needed
if [ ! -f "railway.toml" ]; then
    echo "🎯 Initializing Railway project..."
    railway init
fi

# Deploy to Railway
echo "🚀 Deploying quantum platform..."
railway up --detach

# Get deployment URL
RAILWAY_URL=$(railway domain)

echo ""
echo "🎉 DEPLOYMENT SUCCESSFUL! 🎉"
echo "============================="
echo ""
echo "🌐 Your Quantum Platform is live at:"
echo "    $RAILWAY_URL"
echo ""
echo "📊 Health Check:"
echo "    curl $RAILWAY_URL/health"
echo ""
echo "📚 API Documentation:"
echo "    $RAILWAY_URL/docs"
echo ""
echo "⚡ Quantum Operations:"
echo "    $RAILWAY_URL/quantum/simulate"
echo "    $RAILWAY_URL/quantum/algorithms"
echo "    $RAILWAY_URL/quantum/performance"
echo ""
echo "💰 Ready for customer demos and revenue generation!"
echo ""
echo "🚀 Next Steps:"
echo "   1. Test the /health endpoint"
echo "   2. Explore the /docs interface"
echo "   3. Share the live demo with customers"
echo "   4. Configure custom domain in Railway dashboard"
echo ""
echo "🌟 QUANTUM EMPIRE IS NOW LIVE ON RAILWAY! 🌟" 