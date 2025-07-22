// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Script, console2} from "forge-std/Script.sol";
import {QuantumAlgorithmNFT} from "../src/QuantumAlgorithmNFT.sol";

/**
 * @title DeployQuantumNFT
 * @dev Foundry deployment script for Quantum Algorithm NFT Collection
 * @notice Deploys and configures the quantum NFT contract with production settings
 */
contract DeployQuantumNFT is Script {
    // Configuration
    address public constant DEFAULT_ROYALTY_RECIPIENT = 0x70997970C51812dc3A010C7d01b50e0d17dc79C8;
    bool public constant ACTIVATE_MINTING_ON_DEPLOY = true;
    
    // Deployment tracking
    QuantumAlgorithmNFT public quantumNFT;
    address public deployedAddress;
    uint256 public deploymentBlock;
    
    function run() external {
        // Get configuration from environment
        address royaltyRecipient = getRoyaltyRecipient();
        bool activateMinting = getActivateMinting();
        
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        
        console2.log("===============================================");
        console2.log("🚀 QUANTUM NFT DEPLOYMENT STARTING");
        console2.log("===============================================");
        console2.log("Chain ID:", block.chainid);
        console2.log("Deployer:", vm.addr(deployerPrivateKey));
        console2.log("Royalty Recipient:", royaltyRecipient);
        console2.log("Activate Minting:", activateMinting);
        console2.log("===============================================");
        
        vm.startBroadcast(deployerPrivateKey);
        
        // Deploy contract
        quantumNFT = new QuantumAlgorithmNFT(royaltyRecipient);
        deployedAddress = address(quantumNFT);
        deploymentBlock = block.number;
        
        console2.log("✅ Contract deployed at:", deployedAddress);
        console2.log("📦 Deployment block:", deploymentBlock);
        
        // Configure contract if needed
        if (activateMinting) {
            quantumNFT.setMintingActive(true);
            console2.log("⚡ Minting activated");
        }
        
        // Verify deployment
        verifyDeployment(royaltyRecipient);
        
        vm.stopBroadcast();
        
        // Log deployment summary
        logDeploymentSummary();
        
        // Export deployment data
        exportDeploymentData();
    }
    
    function getRoyaltyRecipient() internal view returns (address) {
        try vm.envAddress("ROYALTY_RECIPIENT") returns (address addr) {
            require(addr != address(0), "Invalid royalty recipient");
            return addr;
        } catch {
            console2.log("⚠️  Using default royalty recipient");
            return DEFAULT_ROYALTY_RECIPIENT;
        }
    }
    
    function getActivateMinting() internal view returns (bool) {
        try vm.envBool("ACTIVATE_MINTING") returns (bool activate) {
            return activate;
        } catch {
            return ACTIVATE_MINTING_ON_DEPLOY;
        }
    }
    
    function verifyDeployment(address expectedRoyaltyRecipient) internal view {
        console2.log("🔍 Verifying deployment...");
        
        // Check basic contract properties
        require(bytes(quantumNFT.name()).length > 0, "Contract name not set");
        require(bytes(quantumNFT.symbol()).length > 0, "Contract symbol not set");
        require(quantumNFT.MAX_SUPPLY() == 10000, "Incorrect max supply");
        require(quantumNFT.MINT_PRICE() == 0.08 ether, "Incorrect mint price");
        require(quantumNFT.ROYALTY_BPS() == 750, "Incorrect royalty BPS");
        
        // Check royalty recipient
        require(quantumNFT.royaltyRecipient() == expectedRoyaltyRecipient, "Incorrect royalty recipient");
        
        // Check total supply starts at 0
        require(quantumNFT.totalSupply() == 0, "Total supply should start at 0");
        
        // Check admin role
        require(
            quantumNFT.hasRole(quantumNFT.DEFAULT_ADMIN_ROLE(), vm.addr(vm.envUint("PRIVATE_KEY"))),
            "Deployer should have admin role"
        );
        
        console2.log("✅ Deployment verification passed");
    }
    
    function logDeploymentSummary() internal view {
        console2.log("");
        console2.log("🎊 QUANTUM NFT DEPLOYMENT COMPLETE! 🎊");
        console2.log("=====================================");
        console2.log("📋 DEPLOYMENT SUMMARY:");
        console2.log("   Contract Address:", deployedAddress);
        console2.log("   Block Number:", deploymentBlock);
        console2.log("   Chain ID:", block.chainid);
        console2.log("   Max Supply:", quantumNFT.MAX_SUPPLY());
        console2.log("   Mint Price:", quantumNFT.MINT_PRICE(), "wei");
        console2.log("   Royalty:", quantumNFT.ROYALTY_BPS(), "bps (7.5%)");
        console2.log("   Minting Active:", quantumNFT.mintingActive());
        console2.log("");
        console2.log("🌟 CONTRACT FEATURES:");
        console2.log("   ✅ ERC721 with Enumerable & URI Storage");
        console2.log("   ✅ Access Control (Admin, Minter, Pauser roles)");
        console2.log("   ✅ Pausable & Reentrancy Protection");
        console2.log("   ✅ EIP-2981 Royalty Standard");
        console2.log("   ✅ Quantum Algorithm Metadata");
        console2.log("   ✅ Platform Access Integration");
        console2.log("");
        console2.log("💰 REVENUE POTENTIAL:");
        console2.log("   Primary Sales: 800 ETH (10,000 × 0.08 ETH)");
        console2.log("   At $2,000 ETH: $1,600,000 revenue");
        console2.log("   Royalties: 7.5% on all secondary sales");
        console2.log("");
        console2.log("🚀 NEXT STEPS:");
        console2.log("   1. Verify contract on Etherscan");
        console2.log("   2. Set up metadata IPFS hosting");
        console2.log("   3. Begin marketing campaign");
        console2.log("   4. Launch public minting");
        console2.log("");
        console2.log("🎯 CONTRACT INTERACTION:");
        
        if (block.chainid == 1) {
            console2.log("   Etherscan: https://etherscan.io/address/", deployedAddress);
        } else if (block.chainid == 11155111) {
            console2.log("   Sepolia Etherscan: https://sepolia.etherscan.io/address/", deployedAddress);
        } else if (block.chainid == 137) {
            console2.log("   Polygonscan: https://polygonscan.com/address/", deployedAddress);
        }
        
        console2.log("");
        console2.log("🌟 QUANTUM NFT REVOLUTION IS LIVE! 🌟");
        console2.log("=====================================");
    }
    
    function exportDeploymentData() internal {
        string memory chainName = getChainName();
        
        // Create deployment data JSON
        string memory json = string.concat(
            '{\n',
            '  "contractName": "QuantumAlgorithmNFT",\n',
            '  "contractAddress": "', vm.toString(deployedAddress), '",\n',
            '  "deployer": "', vm.toString(vm.addr(vm.envUint("PRIVATE_KEY"))), '",\n',
            '  "chainId": ', vm.toString(block.chainid), ',\n',
            '  "chainName": "', chainName, '",\n',
            '  "blockNumber": ', vm.toString(deploymentBlock), ',\n',
            '  "timestamp": ', vm.toString(block.timestamp), ',\n',
            '  "maxSupply": ', vm.toString(quantumNFT.MAX_SUPPLY()), ',\n',
            '  "mintPrice": "', vm.toString(quantumNFT.MINT_PRICE()), '",\n',
            '  "royaltyBPS": ', vm.toString(quantumNFT.ROYALTY_BPS()), ',\n',
            '  "royaltyRecipient": "', vm.toString(quantumNFT.royaltyRecipient()), '",\n',
            '  "mintingActive": ', quantumNFT.mintingActive() ? 'true' : 'false', '\n',
            '}'
        );
        
        // Write deployment data to file
        string memory filename = string.concat("deployments/", chainName, "_deployment.json");
        vm.writeFile(filename, json);
        
        console2.log("📊 Deployment data exported to:", filename);
    }
    
    function getChainName() internal view returns (string memory) {
        if (block.chainid == 1) return "mainnet";
        if (block.chainid == 11155111) return "sepolia";
        if (block.chainid == 137) return "polygon";
        if (block.chainid == 80001) return "mumbai";
        if (block.chainid == 42161) return "arbitrum";
        return "unknown";
    }
    
    // Helper function for testing deployment
    function deployForTesting() external returns (address) {
        quantumNFT = new QuantumAlgorithmNFT(DEFAULT_ROYALTY_RECIPIENT);
        quantumNFT.setMintingActive(true);
        return address(quantumNFT);
    }
} 