// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Test, console2} from "forge-std/Test.sol";
import {QuantumAlgorithmNFT} from "../src/QuantumAlgorithmNFT.sol";
import {IERC721} from "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import {IERC2981} from "@openzeppelin/contracts/interfaces/IERC2981.sol";

/**
 * @title QuantumAlgorithmNFT Test Suite
 * @dev Comprehensive testing for quantum algorithm NFT collection
 */
contract QuantumAlgorithmNFTTest is Test {
    QuantumAlgorithmNFT public nft;
    
    address public owner;
    address public user1;
    address public user2;
    address public royaltyRecipient;
    address public minter;
    address public pauser;
    
    // Test constants
    uint256 public constant MINT_PRICE = 0.08 ether;
    uint256 public constant MAX_SUPPLY = 10000;
    uint256 public constant ROYALTY_BPS = 750; // 7.5%
    
    // Events to test
    event AlgorithmMinted(
        address indexed to, 
        uint256 indexed tokenId, 
        string algorithmType, 
        uint256 quantumAdvantage
    );
    event PlatformAccessGranted(address indexed user, uint256 expiryTimestamp);
    event QuantumAdvantageUpdated(uint256 indexed tokenId, uint256 newAdvantage);

    function setUp() public {
        // Set up test accounts
        owner = makeAddr("owner");
        user1 = makeAddr("user1");
        user2 = makeAddr("user2");
        royaltyRecipient = makeAddr("royaltyRecipient");
        minter = makeAddr("minter");
        pauser = makeAddr("pauser");
        
        // Deploy contract as owner
        vm.startPrank(owner);
        nft = new QuantumAlgorithmNFT(royaltyRecipient);
        
        // Grant roles for testing
        nft.grantRole(nft.MINTER_ROLE(), minter);
        nft.grantRole(nft.PAUSER_ROLE(), pauser);
        
        vm.stopPrank();
        
        // Fund users for testing
        vm.deal(user1, 10 ether);
        vm.deal(user2, 10 ether);
    }

    /*//////////////////////////////////////////////////////////////
                         DEPLOYMENT TESTS
    //////////////////////////////////////////////////////////////*/

    function testDeployment() public {
        assertEq(nft.name(), "Quantum Algorithm Collection");
        assertEq(nft.symbol(), "QUANTUM");
        assertEq(nft.MAX_SUPPLY(), MAX_SUPPLY);
        assertEq(nft.MINT_PRICE(), MINT_PRICE);
        assertEq(nft.ROYALTY_BPS(), ROYALTY_BPS);
        assertEq(nft.royaltyRecipient(), royaltyRecipient);
        assertEq(nft.totalSupply(), 0);
        assertEq(nft.mintingActive(), false);
        
        // Check roles
        assertTrue(nft.hasRole(nft.DEFAULT_ADMIN_ROLE(), owner));
        assertTrue(nft.hasRole(nft.MINTER_ROLE(), owner));
        assertTrue(nft.hasRole(nft.PAUSER_ROLE(), owner));
        assertTrue(nft.hasRole(nft.PLATFORM_ADMIN_ROLE(), owner));
    }

    /*//////////////////////////////////////////////////////////////
                         MINTING TESTS
    //////////////////////////////////////////////////////////////*/

    function testMintAlgorithm() public {
        vm.startPrank(minter);
        
        vm.expectEmit(true, true, false, true);
        emit AlgorithmMinted(user1, 0, "Shor", 1000);
        
        vm.expectEmit(true, false, false, true);
        emit PlatformAccessGranted(user1, block.timestamp + 365 days);
        
        nft.mintAlgorithm(
            user1,
            "Shor",
            1000,
            "BQP",
            4, // Legendary
            2048
        );
        
        vm.stopPrank();
        
        // Verify mint
        assertEq(nft.totalSupply(), 1);
        assertEq(nft.ownerOf(0), user1);
        assertEq(nft.balanceOf(user1), 1);
        
        // Check algorithm data
        QuantumAlgorithmNFT.QuantumAlgorithm memory algo = nft.getAlgorithm(0);
        assertEq(algo.algorithmType, "Shor");
        assertEq(algo.quantumAdvantage, 1000);
        assertEq(algo.complexityClass, "BQP");
        assertEq(algo.rarity, 4);
        assertEq(algo.qubitsRequired, 2048);
        assertTrue(algo.platformAccess);
        assertEq(algo.mintTimestamp, block.timestamp);
        
        // Check platform access
        assertTrue(nft.hasPlatformAccess(user1));
        assertEq(nft.platformAccessExpiry(user1), block.timestamp + 365 days);
        
        // Check rarity supply
        assertEq(nft.raritySupply(4), 1);
    }
    
    function testPublicMint() public {
        // Activate minting
        vm.prank(owner);
        nft.setMintingActive(true);
        
        vm.startPrank(user1);
        
        vm.expectEmit(true, true, false, true);
        emit AlgorithmMinted(user1, 0, "Grover", 4);
        
        nft.publicMint{value: MINT_PRICE}(
            "Grover",
            4,
            "BQP",
            1, // Common
            100
        );
        
        vm.stopPrank();
        
        // Verify mint
        assertEq(nft.totalSupply(), 1);
        assertEq(nft.ownerOf(0), user1);
        assertTrue(nft.hasPlatformAccess(user1));
    }
    
    function testPublicMintWithRefund() public {
        vm.prank(owner);
        nft.setMintingActive(true);
        
        uint256 overpayment = MINT_PRICE + 0.02 ether;
        uint256 balanceBefore = user1.balance;
        
        vm.prank(user1);
        nft.publicMint{value: overpayment}(
            "VQE",
            25,
            "BQP",
            2, // Rare
            50
        );
        
        // Should refund excess
        assertEq(user1.balance, balanceBefore - MINT_PRICE);
    }

    /*//////////////////////////////////////////////////////////////
                         ACCESS CONTROL TESTS
    //////////////////////////////////////////////////////////////*/

    function testOnlyMinterCanMint() public {
        vm.expectRevert();
        vm.prank(user1);
        nft.mintAlgorithm(user1, "Test", 1, "P", 1, 10);
    }
    
    function testOnlyAdminCanSetMinting() public {
        vm.expectRevert();
        vm.prank(user1);
        nft.setMintingActive(true);
        
        // Should work for admin
        vm.prank(owner);
        nft.setMintingActive(true);
        assertTrue(nft.mintingActive());
    }

    /*//////////////////////////////////////////////////////////////
                         VALIDATION TESTS
    //////////////////////////////////////////////////////////////*/

    function testCannotMintToZeroAddress() public {
        vm.expectRevert("Cannot mint to zero address");
        vm.prank(minter);
        nft.mintAlgorithm(address(0), "Test", 1, "P", 1, 10);
    }
    
    function testCannotMintWithInvalidRarity() public {
        vm.expectRevert("Invalid rarity");
        vm.prank(minter);
        nft.mintAlgorithm(user1, "Test", 1, "P", 0, 10);
        
        vm.expectRevert("Invalid rarity");
        vm.prank(minter);
        nft.mintAlgorithm(user1, "Test", 1, "P", 6, 10);
    }
    
    function testCannotMintWithZeroAdvantage() public {
        vm.expectRevert("Quantum advantage must be positive");
        vm.prank(minter);
        nft.mintAlgorithm(user1, "Test", 0, "P", 1, 10);
    }
    
    function testCannotMintWithEmptyAlgorithmType() public {
        vm.expectRevert("Algorithm type required");
        vm.prank(minter);
        nft.mintAlgorithm(user1, "", 1, "P", 1, 10);
    }

    /*//////////////////////////////////////////////////////////////
                         MAX SUPPLY TESTS
    //////////////////////////////////////////////////////////////*/

    function testCannotExceedMaxSupply() public {
        // Set token counter near max (using cheat code for testing)
        vm.store(address(nft), bytes32(uint256(8)), bytes32(uint256(MAX_SUPPLY)));
        
        vm.expectRevert("Max supply exceeded");
        vm.prank(minter);
        nft.mintAlgorithm(user1, "Test", 1, "P", 1, 10);
    }

    /*//////////////////////////////////////////////////////////////
                         PUBLIC MINT VALIDATION TESTS
    //////////////////////////////////////////////////////////////*/

    function testPublicMintRequiresMintingActive() public {
        vm.expectRevert("Minting not active");
        vm.prank(user1);
        nft.publicMint{value: MINT_PRICE}("Test", 1, "P", 1, 10);
    }
    
    function testPublicMintRequiresSufficientPayment() public {
        vm.prank(owner);
        nft.setMintingActive(true);
        
        vm.expectRevert("Insufficient payment");
        vm.prank(user1);
        nft.publicMint{value: MINT_PRICE - 1}("Test", 1, "P", 1, 10);
    }

    /*//////////////////////////////////////////////////////////////
                         PLATFORM ACCESS TESTS
    //////////////////////////////////////////////////////////////*/

    function testPlatformAccessExpiry() public {
        vm.prank(minter);
        nft.mintAlgorithm(user1, "Test", 1, "P", 1, 10);
        
        // Should have access initially
        assertTrue(nft.hasPlatformAccess(user1));
        
        // Jump forward 366 days
        vm.warp(block.timestamp + 366 days);
        
        // Should no longer have access
        assertFalse(nft.hasPlatformAccess(user1));
    }

    /*//////////////////////////////////////////////////////////////
                         ALGORITHM UPDATE TESTS
    //////////////////////////////////////////////////////////////*/

    function testUpdateQuantumAdvantage() public {
        vm.prank(minter);
        nft.mintAlgorithm(user1, "Test", 1, "P", 1, 10);
        
        vm.expectEmit(true, false, false, true);
        emit QuantumAdvantageUpdated(0, 100);
        
        vm.prank(owner); // Platform admin
        nft.updateQuantumAdvantage(0, 100);
        
        QuantumAlgorithmNFT.QuantumAlgorithm memory algo = nft.getAlgorithm(0);
        assertEq(algo.quantumAdvantage, 100);
    }
    
    function testCannotUpdateNonexistentToken() public {
        vm.expectRevert("Token does not exist");
        vm.prank(owner);
        nft.updateQuantumAdvantage(999, 100);
    }
    
    function testCannotUpdateToZeroAdvantage() public {
        vm.prank(minter);
        nft.mintAlgorithm(user1, "Test", 1, "P", 1, 10);
        
        vm.expectRevert("Advantage must be positive");
        vm.prank(owner);
        nft.updateQuantumAdvantage(0, 0);
    }

    /*//////////////////////////////////////////////////////////////
                         PAUSE TESTS
    //////////////////////////////////////////////////////////////*/

    function testPauseAndUnpause() public {
        vm.prank(pauser);
        nft.pause();
        assertTrue(nft.paused());
        
        // Should prevent minting
        vm.expectRevert("Pausable: paused");
        vm.prank(minter);
        nft.mintAlgorithm(user1, "Test", 1, "P", 1, 10);
        
        vm.prank(pauser);
        nft.unpause();
        assertFalse(nft.paused());
        
        // Should allow minting again
        vm.prank(minter);
        nft.mintAlgorithm(user1, "Test", 1, "P", 1, 10);
        assertEq(nft.totalSupply(), 1);
    }

    /*//////////////////////////////////////////////////////////////
                         ROYALTY TESTS
    //////////////////////////////////////////////////////////////*/

    function testRoyaltyInfo() public {
        vm.prank(minter);
        nft.mintAlgorithm(user1, "Test", 1, "P", 1, 10);
        
        uint256 salePrice = 1 ether;
        (address receiver, uint256 royaltyAmount) = nft.royaltyInfo(0, salePrice);
        
        assertEq(receiver, royaltyRecipient);
        assertEq(royaltyAmount, (salePrice * ROYALTY_BPS) / 10000);
    }
    
    function testRoyaltyInfoNonexistentToken() public {
        vm.expectRevert("Token does not exist");
        nft.royaltyInfo(999, 1 ether);
    }

    /*//////////////////////////////////////////////////////////////
                         WITHDRAWAL TESTS
    //////////////////////////////////////////////////////////////*/

    function testWithdraw() public {
        vm.prank(owner);
        nft.setMintingActive(true);
        
        // Mint to generate revenue
        vm.prank(user1);
        nft.publicMint{value: MINT_PRICE}("Test", 1, "P", 1, 10);
        
        uint256 balanceBefore = royaltyRecipient.balance;
        
        vm.prank(owner);
        nft.withdraw();
        
        assertEq(royaltyRecipient.balance, balanceBefore + MINT_PRICE);
        assertEq(address(nft).balance, 0);
    }
    
    function testCannotWithdrawWithZeroBalance() public {
        vm.expectRevert("No funds to withdraw");
        vm.prank(owner);
        nft.withdraw();
    }

    /*//////////////////////////////////////////////////////////////
                         INTERFACE SUPPORT TESTS
    //////////////////////////////////////////////////////////////*/

    function testSupportsInterface() public {
        assertTrue(nft.supportsInterface(type(IERC721).interfaceId));
        assertTrue(nft.supportsInterface(type(IERC2981).interfaceId));
    }

    /*//////////////////////////////////////////////////////////////
                         FUZZ TESTS
    //////////////////////////////////////////////////////////////*/

    function testFuzzMintAlgorithm(
        string calldata algorithmType,
        uint256 quantumAdvantage,
        string calldata complexityClass,
        uint8 rarity,
        uint256 qubitsRequired
    ) public {
        // Bound inputs to valid ranges
        vm.assume(bytes(algorithmType).length > 0);
        vm.assume(quantumAdvantage > 0);
        rarity = uint8(bound(rarity, 1, 5));
        
        vm.prank(minter);
        nft.mintAlgorithm(
            user1,
            algorithmType,
            quantumAdvantage,
            complexityClass,
            rarity,
            qubitsRequired
        );
        
        QuantumAlgorithmNFT.QuantumAlgorithm memory algo = nft.getAlgorithm(0);
        assertEq(algo.algorithmType, algorithmType);
        assertEq(algo.quantumAdvantage, quantumAdvantage);
        assertEq(algo.complexityClass, complexityClass);
        assertEq(algo.rarity, rarity);
        assertEq(algo.qubitsRequired, qubitsRequired);
    }

    /*//////////////////////////////////////////////////////////////
                         INTEGRATION TESTS
    //////////////////////////////////////////////////////////////*/

    function testMintMultipleAlgorithms() public {
        vm.startPrank(minter);
        
        // Mint different algorithm types
        nft.mintAlgorithm(user1, "Shor", 1000, "BQP", 4, 2048);
        nft.mintAlgorithm(user1, "Grover", 4, "BQP", 1, 100);
        nft.mintAlgorithm(user2, "VQE", 25, "BQP", 2, 50);
        
        vm.stopPrank();
        
        assertEq(nft.totalSupply(), 3);
        assertEq(nft.balanceOf(user1), 2);
        assertEq(nft.balanceOf(user2), 1);
        
        // Check rarity tracking
        assertEq(nft.raritySupply(1), 1); // Common
        assertEq(nft.raritySupply(2), 1); // Rare
        assertEq(nft.raritySupply(4), 1); // Legendary
    }

    /*//////////////////////////////////////////////////////////////
                         GAS OPTIMIZATION TESTS
    //////////////////////////////////////////////////////////////*/

    function testGasOptimizedMinting() public {
        vm.prank(owner);
        nft.setMintingActive(true);
        
        uint256 gasBefore = gasleft();
        
        vm.prank(user1);
        nft.publicMint{value: MINT_PRICE}("Test", 1, "P", 1, 10);
        
        uint256 gasUsed = gasBefore - gasleft();
        
        // Gas should be reasonable (under 200k for complex mint)
        assertLt(gasUsed, 200000);
        
        console2.log("Gas used for public mint:", gasUsed);
    }
} 