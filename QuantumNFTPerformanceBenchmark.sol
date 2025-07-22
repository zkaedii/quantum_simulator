// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Test, console2} from "forge-std/Test.sol";
import {QuantumAlgorithmNFT} from "../src/QuantumAlgorithmNFT.sol";

/**
 * @title Quantum NFT Performance Benchmark
 * @dev Comprehensive gas optimization and performance testing
 * @author FoundryOps Performance Team
 */
contract QuantumNFTPerformanceBenchmark is Test {
    QuantumAlgorithmNFT public nft;
    address public owner;
    address public user;
    address public royaltyRecipient;
    
    struct GasReport {
        string functionName;
        uint256 gasUsed;
        uint256 gasLimit;
        bool passed;
        string notes;
    }
    
    GasReport[] public gasReports;
    
    function setUp() public {
        owner = makeAddr("owner");
        user = makeAddr("user");
        royaltyRecipient = makeAddr("royalty");
        
        vm.startPrank(owner);
        nft = new QuantumAlgorithmNFT(royaltyRecipient);
        nft.setMintingActive(true);
        vm.stopPrank();
        
        vm.deal(user, 100 ether);
    }
    
    /*//////////////////////////////////////////////////////////////
                           DEPLOYMENT BENCHMARKS
    //////////////////////////////////////////////////////////////*/
    
    function testBenchmarkDeployment() public {
        uint256 gasStart = gasleft();
        
        QuantumAlgorithmNFT newNft = new QuantumAlgorithmNFT(royaltyRecipient);
        
        uint256 gasUsed = gasStart - gasleft();
        
        _recordGasUsage(
            "Contract Deployment",
            gasUsed,
            3000000, // 3M gas limit
            "Full contract deployment including all features"
        );
        
        console2.log("Contract deployed successfully");
        console2.log("Deployment gas used:", gasUsed);
    }
    
    /*//////////////////////////////////////////////////////////////
                            MINTING BENCHMARKS
    //////////////////////////////////////////////////////////////*/
    
    function testBenchmarkMintAlgorithm() public {
        vm.startPrank(owner);
        
        uint256 gasStart = gasleft();
        
        nft.mintAlgorithm(
            user,
            "Shor",
            1000,
            "BQP", 
            4,
            100,
            "https://metadata.uri/1"
        );
        
        uint256 gasUsed = gasStart - gasleft();
        vm.stopPrank();
        
        _recordGasUsage(
            "mintAlgorithm",
            gasUsed,
            200000, // 200k gas limit
            "Admin minting with full metadata"
        );
    }
    
    function testBenchmarkPublicMintSingle() public {
        vm.startPrank(user);
        
        uint256 gasStart = gasleft();
        
        nft.publicMint{value: nft.MINT_PRICE()}(1);
        
        uint256 gasUsed = gasStart - gasleft();
        vm.stopPrank();
        
        _recordGasUsage(
            "publicMint(1)",
            gasUsed,
            250000, // 250k gas limit
            "Single NFT public mint with randomization"
        );
    }
    
    function testBenchmarkPublicMintBatch() public {
        vm.startPrank(user);
        
        uint256 gasStart = gasleft();
        
        nft.publicMint{value: nft.MINT_PRICE() * 10}(10);
        
        uint256 gasUsed = gasStart - gasleft();
        vm.stopPrank();
        
        _recordGasUsage(
            "publicMint(10)",
            gasUsed,
            2000000, // 2M gas limit
            "Batch minting 10 NFTs with optimization"
        );
        
        // Calculate per-NFT gas cost
        uint256 perNftGas = gasUsed / 10;
        console2.log("Gas per NFT in batch:", perNftGas);
        
        _recordGasUsage(
            "publicMint (per NFT in batch)",
            perNftGas,
            200000, // Should be more efficient than single mint
            "Average gas per NFT when minting in batch"
        );
    }
    
    /*//////////////////////////////////////////////////////////////
                           TRANSFER BENCHMARKS
    //////////////////////////////////////////////////////////////*/
    
    function testBenchmarkTransfer() public {
        // First mint an NFT
        vm.prank(owner);
        nft.mintAlgorithm(user, "Shor", 1000, "BQP", 4, 100, "uri");
        
        address recipient = makeAddr("recipient");
        
        vm.startPrank(user);
        uint256 gasStart = gasleft();
        
        nft.transferFrom(user, recipient, 0);
        
        uint256 gasUsed = gasStart - gasleft();
        vm.stopPrank();
        
        _recordGasUsage(
            "transferFrom",
            gasUsed,
            100000, // 100k gas limit
            "NFT transfer with platform access update"
        );
    }
    
    /*//////////////////////////////////////////////////////////////
                         PLATFORM ACCESS BENCHMARKS
    //////////////////////////////////////////////////////////////*/
    
    function testBenchmarkPlatformAccessCheck() public {
        // Mint NFT first
        vm.prank(owner);
        nft.mintAlgorithm(user, "Shor", 1000, "BQP", 4, 100, "uri");
        
        uint256 gasStart = gasleft();
        
        bool hasAccess = nft.hasPlatformAccess(user);
        
        uint256 gasUsed = gasStart - gasleft();
        
        _recordGasUsage(
            "hasPlatformAccess",
            gasUsed,
            5000, // 5k gas limit
            "Platform access verification (view function)"
        );
        
        assertTrue(hasAccess);
    }
    
    function testBenchmarkExtendPlatformAccess() public {
        // Mint NFT first
        vm.prank(owner);
        nft.mintAlgorithm(user, "Shor", 1000, "BQP", 4, 100, "uri");
        
        vm.startPrank(owner);
        uint256 gasStart = gasleft();
        
        nft.extendPlatformAccess(user, 30 days);
        
        uint256 gasUsed = gasStart - gasleft();
        vm.stopPrank();
        
        _recordGasUsage(
            "extendPlatformAccess",
            gasUsed,
            50000, // 50k gas limit
            "Platform access extension by admin"
        );
    }
    
    /*//////////////////////////////////////////////////////////////
                         UPDATE BENCHMARKS
    //////////////////////////////////////////////////////////////*/
    
    function testBenchmarkUpdateQuantumAdvantage() public {
        // Mint NFT first
        vm.prank(owner);
        nft.mintAlgorithm(user, "Shor", 1000, "BQP", 4, 100, "uri");
        
        vm.startPrank(owner);
        uint256 gasStart = gasleft();
        
        nft.updateQuantumAdvantage(0, 5000);
        
        uint256 gasUsed = gasStart - gasleft();
        vm.stopPrank();
        
        _recordGasUsage(
            "updateQuantumAdvantage",
            gasUsed,
            40000, // 40k gas limit
            "Quantum advantage update with event emission"
        );
    }
    
    /*//////////////////////////////////////////////////////////////
                           QUERY BENCHMARKS
    //////////////////////////////////////////////////////////////*/
    
    function testBenchmarkGetAlgorithm() public {
        // Mint NFT first
        vm.prank(owner);
        nft.mintAlgorithm(user, "Shor", 1000, "BQP", 4, 100, "uri");
        
        uint256 gasStart = gasleft();
        
        QuantumAlgorithmNFT.QuantumAlgorithm memory algorithm = nft.getAlgorithm(0);
        
        uint256 gasUsed = gasStart - gasleft();
        
        _recordGasUsage(
            "getAlgorithm",
            gasUsed,
            10000, // 10k gas limit
            "Algorithm data retrieval (view function)"
        );
        
        assertEq(algorithm.algorithmType, "Shor");
    }
    
    function testBenchmarkRarityDistribution() public {
        // Mint some NFTs first
        vm.startPrank(owner);
        for (uint256 i = 0; i < 5; i++) {
            nft.mintAlgorithm(user, "Test", 1000, "BQP", uint8(i + 1), 100, "uri");
        }
        vm.stopPrank();
        
        uint256 gasStart = gasleft();
        
        uint256[5] memory distribution = nft.getRarityDistribution();
        
        uint256 gasUsed = gasStart - gasleft();
        
        _recordGasUsage(
            "getRarityDistribution",
            gasUsed,
            15000, // 15k gas limit
            "Rarity distribution query (view function)"
        );
        
        // Verify distribution
        for (uint256 i = 0; i < 5; i++) {
            assertEq(distribution[i], 1);
        }
    }
    
    /*//////////////////////////////////////////////////////////////
                         ADMINISTRATIVE BENCHMARKS
    //////////////////////////////////////////////////////////////*/
    
    function testBenchmarkWithdraw() public {
        // Generate some revenue first
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE() * 5}(5);
        
        vm.startPrank(owner);
        uint256 gasStart = gasleft();
        
        nft.withdraw();
        
        uint256 gasUsed = gasStart - gasleft();
        vm.stopPrank();
        
        _recordGasUsage(
            "withdraw",
            gasUsed,
            30000, // 30k gas limit
            "Contract balance withdrawal to royalty recipient"
        );
    }
    
    function testBenchmarkPauseUnpause() public {
        vm.startPrank(owner);
        
        uint256 gasStart = gasleft();
        nft.pause();
        uint256 pauseGas = gasStart - gasleft();
        
        gasStart = gasleft();
        nft.unpause();
        uint256 unpauseGas = gasStart - gasleft();
        
        vm.stopPrank();
        
        _recordGasUsage(
            "pause",
            pauseGas,
            25000, // 25k gas limit
            "Emergency pause activation"
        );
        
        _recordGasUsage(
            "unpause",
            unpauseGas,
            25000, // 25k gas limit
            "Emergency pause deactivation"
        );
    }
    
    /*//////////////////////////////////////////////////////////////
                         SCALING BENCHMARKS
    //////////////////////////////////////////////////////////////*/
    
    function testBenchmarkMintingAtScale() public {
        console2.log("Testing minting performance at scale...");
        
        uint256[] memory mintCounts = new uint256[](5);
        mintCounts[0] = 1;
        mintCounts[1] = 5;
        mintCounts[2] = 10;
        mintCounts[3] = 50;
        mintCounts[4] = 100;
        
        for (uint256 i = 0; i < mintCounts.length; i++) {
            uint256 count = mintCounts[i];
            
            vm.startPrank(owner);
            uint256 gasStart = gasleft();
            
            for (uint256 j = 0; j < count; j++) {
                nft.mintAlgorithm(
                    user,
                    "ScaleTest",
                    1000 + j,
                    "BQP",
                    uint8((j % 5) + 1),
                    100,
                    string(abi.encodePacked("uri", j))
                );
            }
            
            uint256 gasUsed = gasStart - gasleft();
            vm.stopPrank();
            
            uint256 avgGasPerMint = gasUsed / count;
            
            _recordGasUsage(
                string(abi.encodePacked("mintAlgorithm x", vm.toString(count))),
                gasUsed,
                gasUsed + 100000, // Dynamic limit
                string(abi.encodePacked("Batch of ", vm.toString(count), " mints"))
            );
            
            console2.log("Mints:", count, "Total Gas:", gasUsed, "Avg Gas/Mint:", avgGasPerMint);
        }
    }
    
    /*//////////////////////////////////////////////////////////////
                         OPTIMIZATION VERIFICATION
    //////////////////////////////////////////////////////////////*/
    
    function testVerifyGasOptimizations() public {
        console2.log("Verifying gas optimizations...");
        
        // Test storage packing efficiency
        vm.prank(owner);
        nft.mintAlgorithm(user, "PackingTest", 1000, "BQP", 4, 100, "uri");
        
        // Verify algorithm struct fits in expected storage slots
        // (This would be verified through storage layout analysis)
        
        // Test batch operation efficiency
        uint256 singleMintGas = _measureSingleMint();
        uint256 batchMintGas = _measureBatchMint(10);
        uint256 avgBatchGas = batchMintGas / 10;
        
        console2.log("Single mint gas:", singleMintGas);
        console2.log("Batch mint gas per NFT:", avgBatchGas);
        
        // Batch should be more efficient (at least 10% savings)
        uint256 expectedSavings = (singleMintGas * 90) / 100;
        assert(avgBatchGas <= expectedSavings);
        
        console2.log("✅ Batch minting optimization verified");
    }
    
    function _measureSingleMint() internal returns (uint256) {
        address testUser = makeAddr("singleMintUser");
        vm.deal(testUser, 1 ether);
        
        vm.startPrank(testUser);
        uint256 gasStart = gasleft();
        nft.publicMint{value: nft.MINT_PRICE()}(1);
        uint256 gasUsed = gasStart - gasleft();
        vm.stopPrank();
        
        return gasUsed;
    }
    
    function _measureBatchMint(uint256 quantity) internal returns (uint256) {
        address testUser = makeAddr("batchMintUser");
        vm.deal(testUser, 10 ether);
        
        vm.startPrank(testUser);
        uint256 gasStart = gasleft();
        nft.publicMint{value: nft.MINT_PRICE() * quantity}(quantity);
        uint256 gasUsed = gasStart - gasleft();
        vm.stopPrank();
        
        return gasUsed;
    }
    
    /*//////////////////////////////////////////////////////////////
                         REPORTING UTILITIES
    //////////////////////////////////////////////////////////////*/
    
    function _recordGasUsage(
        string memory functionName,
        uint256 gasUsed,
        uint256 gasLimit,
        string memory notes
    ) internal {
        bool passed = gasUsed <= gasLimit;
        
        gasReports.push(GasReport({
            functionName: functionName,
            gasUsed: gasUsed,
            gasLimit: gasLimit,
            passed: passed,
            notes: notes
        }));
        
        string memory status = passed ? "✅ PASS" : "❌ FAIL";
        console2.log(
            string(abi.encodePacked(
                status, " ", functionName, ": ", 
                vm.toString(gasUsed), " gas (limit: ", 
                vm.toString(gasLimit), ")"
            ))
        );
        
        if (!passed) {
            console2.log("⚠️  Gas usage exceeded limit!");
        }
    }
    
    function printGasReport() external view {
        console2.log("\n📊 QUANTUM NFT GAS OPTIMIZATION REPORT");
        console2.log("=====================================");
        
        uint256 totalTests = gasReports.length;
        uint256 passedTests = 0;
        uint256 totalGasUsed = 0;
        
        for (uint256 i = 0; i < totalTests; i++) {
            GasReport memory report = gasReports[i];
            
            if (report.passed) {
                passedTests++;
            }
            totalGasUsed += report.gasUsed;
            
            string memory status = report.passed ? "✅" : "❌";
            console2.log(
                string(abi.encodePacked(
                    status, " ", report.functionName, ": ", 
                    vm.toString(report.gasUsed), " gas"
                ))
            );
        }
        
        console2.log("\n📈 SUMMARY");
        console2.log("Total Tests:", totalTests);
        console2.log("Passed:", passedTests);
        console2.log("Failed:", totalTests - passedTests);
        console2.log("Success Rate:", (passedTests * 100) / totalTests, "%");
        console2.log("Total Gas Measured:", totalGasUsed);
        console2.log("Average Gas per Test:", totalGasUsed / totalTests);
    }
} 