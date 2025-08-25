#!/bin/bash

# Vercel Deployment Script for Multilingual Voice Assistant

echo "🚀 Deploying Multilingual Voice Assistant to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Check if user is logged in
echo "📝 Checking Vercel authentication..."
vercel whoami || {
    echo "🔐 Please login to Vercel..."
    vercel login
}

# Deploy to Vercel
echo "🚀 Starting deployment..."
vercel --prod

echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Set up environment variables in Vercel dashboard:"
echo "   - GEMINI_API_KEY (required)"
echo "   - DHRUVA_AUTH_TOKEN (optional)"
echo "   - OPENWEATHER_API_KEY (optional)"
echo ""
echo "2. Test your API endpoints:"
echo "   - GET  /customers"
echo ""
echo "3. Check the deployment URL provided by Vercel"
echo ""
echo "🔗 Useful commands:"
echo "   vercel env add GEMINI_API_KEY     # Add environment variable"
echo "   vercel logs                       # View deployment logs"
echo "   vercel domains                    # Manage custom domains"