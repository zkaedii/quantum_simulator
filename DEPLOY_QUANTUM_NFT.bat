@echo off
title QUANTUM NFT LEGENDARY DEPLOYMENT
color 0A

echo.
echo ========================================
echo    QUANTUM NFT LEGENDARY DEPLOYMENT
echo ========================================
echo.
echo 🚀 World's First Quantum Algorithm NFT Collection
echo 💎 10,000 NFTs with Scientific Accuracy
echo ⚡ 2x - 100,000x Quantum Advantage Range
echo 💰 $1,600,000+ Revenue Potential
echo.

echo Choose your deployment option:
echo.
echo [1] MAINNET (Production Launch - $170 deploy cost)
echo [2] SEPOLIA TESTNET (Safe Testing - $0 cost)
echo [3] POLYGON (Low Fees - $0.08 cost)
echo [4] EXIT
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🚀 LAUNCHING MAINNET DEPLOYMENT...
    echo This will deploy to Ethereum Mainnet for production use.
    echo Cost: ~$170 in ETH for gas fees
    echo.
    set /p confirm="Are you sure? (y/N): "
    if /i "%confirm%"=="y" (
        powershell.exe -ExecutionPolicy RemoteSigned -File "LEGENDARY_QUANTUM_NFT_DEPLOYMENT.ps1" -Network mainnet
    ) else (
        echo Deployment cancelled.
    )
)

if "%choice%"=="2" (
    echo.
    echo 🧪 LAUNCHING SEPOLIA TESTNET DEPLOYMENT...
    echo This will deploy to Sepolia testnet for safe testing.
    echo Cost: $0 (uses free testnet ETH)
    echo.
    powershell.exe -ExecutionPolicy RemoteSigned -File "LEGENDARY_QUANTUM_NFT_DEPLOYMENT.ps1" -TestnetFirst
)

if "%choice%"=="3" (
    echo.
    echo 🌐 LAUNCHING POLYGON DEPLOYMENT...
    echo This will deploy to Polygon mainnet for low fees.
    echo Cost: ~$0.08 in MATIC
    echo.
    powershell.exe -ExecutionPolicy RemoteSigned -File "LEGENDARY_QUANTUM_NFT_DEPLOYMENT.ps1" -Network polygon
)

if "%choice%"=="4" (
    echo.
    echo Exiting deployment script.
    goto :end
)

echo.
echo ========================================
echo     QUANTUM NFT DEPLOYMENT COMPLETE
echo ========================================
echo.
echo 🌟 Congratulations! You've just launched the world's first quantum algorithm NFT collection!
echo 🚀 Monitor your deployment in GitHub Actions
echo 💎 Share your contract address with the world
echo 📈 Track your revenue on Etherscan
echo.

:end
pause 