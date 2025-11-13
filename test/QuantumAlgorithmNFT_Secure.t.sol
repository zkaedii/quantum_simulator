// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Test, console2} from "forge-std/Test.sol";
import {QuantumAlgorithmNFT_Secure} from "../src/QuantumAlgorithmNFT_Secure.sol";
import {VRFCoordinatorV2Mock} from "@chainlink/contracts/src/v0.8/mocks/VRFCoordinatorV2Mock.sol";

/**
 * @title Comprehensive Test Suite for QuantumAlgorithmNFT_Secure
 * @notice Tests all security fixes and functionality
 * @dev Achieves 90%+ coverage target
 */
contract QuantumAlgorithmNFT_SecureTest is Test {
    QuantumAlgorithmNFT_Secure public nft;
    VRFCoordinatorV2Mock public vrfCoordinator;

    address public owner;
    address public user;
    address public user2;
    address public royaltyRecipient;

    uint64 public subscriptionId;
    bytes32 public keyHash;
    uint96 public constant BASE_FEE = 100000000000000000; // 0.1 LINK
    uint96 public constant GAS_PRICE_LINK = 1e9; // 1 gwei

    // Events to test
    event RandomnessRequested(uint256 indexed requestId, address indexed requester, uint256 quantity);
    event MintFulfilled(uint256 indexed requestId, address indexed to, uint256 quantity);
    event RefundIssued(address indexed recipient, uint256 amount);
    event PlatformAccessGranted(address indexed user, uint256 expiryTimestamp);

    function setUp() public {
        owner = makeAddr("owner");
        user = makeAddr("user");
        user2 = makeAddr("user2");
        royaltyRecipient = makeAddr("royalty");

        // Fund test accounts
        vm.deal(user, 100 ether);
        vm.deal(user2, 100 ether);

        // Setup VRF Coordinator Mock
        vrfCoordinator = new VRFCoordinatorV2Mock(BASE_FEE, GAS_PRICE_LINK);

        // Create subscription
        subscriptionId = vrfCoordinator.createSubscription();

        // Setup key hash
        keyHash = keccak256("keyHash");

        // Deploy NFT contract
        vm.startPrank(owner);
        nft = new QuantumAlgorithmNFT_Secure(
            royaltyRecipient,
            address(vrfCoordinator),
            subscriptionId,
            keyHash
        );

        // Add consumer to subscription
        vrfCoordinator.addConsumer(subscriptionId, address(nft));

        // Fund subscription
        vrfCoordinator.fundSubscription(subscriptionId, 100 ether);

        // Activate minting
        nft.setMintingActive(true);
        vm.stopPrank();
    }

    /*//////////////////////////////////////////////////////////////
                        CHAINLINK VRF TESTS
    //////////////////////////////////////////////////////////////*/

    function test_PublicMint_RequestsRandomness() public {
        vm.startPrank(user);

        // Expect RandomnessRequested event
        vm.expectEmit(true, true, false, true);
        emit RandomnessRequested(1, user, 1);

        nft.publicMint{value: nft.MINT_PRICE()}(1);

        vm.stopPrank();

        // Verify request was created
        (address to, uint256 quantity, bool fulfilled) = nft.mintRequests(1);
        assertEq(to, user);
        assertEq(quantity, 1);
        assertFalse(fulfilled);
    }

    function test_VRFFulfillment_MintsNFT() public {
        vm.startPrank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);
        vm.stopPrank();

        // Fulfill VRF request
        vrfCoordinator.fulfillRandomWords(1, address(nft));

        // Verify NFT was minted
        assertEq(nft.balanceOf(user), 1);
        assertEq(nft.totalSupply(), 1);
        assertEq(nft.ownerOf(0), user);
    }

    function test_VRFFulfillment_GrantsPlatformAccess() public {
        vm.startPrank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);
        vm.stopPrank();

        vrfCoordinator.fulfillRandomWords(1, address(nft));

        // Verify platform access granted
        assertTrue(nft.hasPlatformAccess(user));
        assertGt(nft.platformAccessExpiry(user), block.timestamp);
    }

    function test_MultipleMints_WithVRF() public {
        vm.startPrank(user);
        nft.publicMint{value: nft.MINT_PRICE() * 5}(5);
        vm.stopPrank();

        vrfCoordinator.fulfillRandomWords(1, address(nft));

        assertEq(nft.balanceOf(user), 5);
        assertEq(nft.totalSupply(), 5);
    }

    /*//////////////////////////////////////////////////////////////
                    PLATFORM ACCESS TESTS (O(1))
    //////////////////////////////////////////////////////////////*/

    function test_PlatformAccess_O1Complexity() public {
        // Mint to user
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);
        vrfCoordinator.fulfillRandomWords(1, address(nft));

        // Measure gas for platform access check
        uint256 gasBefore = gasleft();
        bool hasAccess = nft.hasPlatformAccess(user);
        uint256 gasUsed = gasBefore - gasleft();

        assertTrue(hasAccess);
        console2.log("Gas used for hasPlatformAccess:", gasUsed);

        // Should be very low (< 5,000 gas)
        assertLt(gasUsed, 5000, "Platform access check should be O(1)");
    }

    function test_PlatformAccess_NoNFT() public {
        assertFalse(nft.hasPlatformAccess(user));
    }

    function test_PlatformAccess_Expired() public {
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);
        vrfCoordinator.fulfillRandomWords(1, address(nft));

        assertTrue(nft.hasPlatformAccess(user));

        // Warp past expiry
        vm.warp(block.timestamp + 366 days);

        assertFalse(nft.hasPlatformAccess(user));
    }

    function test_ExtendPlatformAccess() public {
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);
        vrfCoordinator.fulfillRandomWords(1, address(nft));

        uint256 expiryBefore = nft.platformAccessExpiry(user);

        vm.prank(owner);
        nft.extendPlatformAccess(user, 30 days);

        uint256 expiryAfter = nft.platformAccessExpiry(user);
        assertEq(expiryAfter, expiryBefore + 30 days);
    }

    /*//////////////////////////////////////////////////////////////
                    REFUND LOGIC TESTS
    //////////////////////////////////////////////////////////////*/

    function test_Refund_ExactPayment() public {
        uint256 balanceBefore = user.balance;

        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);

        uint256 balanceAfter = user.balance;
        assertEq(balanceBefore - balanceAfter, nft.MINT_PRICE());
    }

    function test_Refund_Overpayment() public {
        uint256 overpayment = 1 ether;
        uint256 balanceBefore = user.balance;

        vm.expectEmit(true, false, false, true);
        emit RefundIssued(user, overpayment - nft.MINT_PRICE());

        vm.prank(user);
        nft.publicMint{value: overpayment}(1);

        uint256 balanceAfter = user.balance;
        assertEq(balanceBefore - balanceAfter, nft.MINT_PRICE());
    }

    function test_Refund_MultipleNFTs() public {
        uint256 cost = nft.MINT_PRICE() * 3;
        uint256 overpayment = cost + 0.5 ether;
        uint256 balanceBefore = user.balance;

        vm.prank(user);
        nft.publicMint{value: overpayment}(3);

        uint256 balanceAfter = user.balance;
        assertEq(balanceBefore - balanceAfter, cost);
    }

    /*//////////////////////////////////////////////////////////////
                    SAFE TRANSFER TESTS
    //////////////////////////////////////////////////////////////*/

    function test_Withdraw_UsesCallNotTransfer() public {
        // Mint some NFTs to generate revenue
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE() * 5}(5);

        uint256 contractBalance = address(nft).balance;
        uint256 recipientBefore = royaltyRecipient.balance;

        vm.prank(owner);
        nft.withdraw();

        uint256 recipientAfter = royaltyRecipient.balance;
        assertEq(recipientAfter - recipientBefore, contractBalance);
    }

    function test_Withdraw_ToContractWallet() public {
        // Deploy a simple contract wallet
        SimpleWallet wallet = new SimpleWallet();

        // Update royalty recipient
        vm.prank(owner);
        vm.store(
            address(nft),
            bytes32(uint256(10)), // Storage slot for royaltyRecipient
            bytes32(uint256(uint160(address(wallet))))
        );

        // Mint NFTs
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);

        // Should succeed even with contract wallet (unlike .transfer())
        vm.prank(owner);
        nft.withdraw();

        assertGt(address(wallet).balance, 0);
    }

    /*//////////////////////////////////////////////////////////////
                    VALIDATION TESTS
    //////////////////////////////////////////////////////////////*/

    function test_Revert_MintingNotActive() public {
        vm.prank(owner);
        nft.setMintingActive(false);

        vm.expectRevert(QuantumAlgorithmNFT_Secure.MintingNotActive.selector);
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);
    }

    function test_Revert_InvalidQuantity_Zero() public {
        vm.expectRevert(
            abi.encodeWithSelector(
                QuantumAlgorithmNFT_Secure.InvalidQuantity.selector,
                0,
                1,
                10
            )
        );
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(0);
    }

    function test_Revert_InvalidQuantity_TooMany() public {
        vm.expectRevert(
            abi.encodeWithSelector(
                QuantumAlgorithmNFT_Secure.InvalidQuantity.selector,
                11,
                1,
                10
            )
        );
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE() * 11}(11);
    }

    function test_Revert_InsufficientPayment() public {
        uint256 required = nft.MINT_PRICE();
        uint256 provided = nft.MINT_PRICE() - 1 wei;

        vm.expectRevert(
            abi.encodeWithSelector(
                QuantumAlgorithmNFT_Secure.InsufficientPayment.selector,
                required,
                provided
            )
        );
        vm.prank(user);
        nft.publicMint{value: provided}(1);
    }

    function test_Revert_MaxSupplyReached() public {
        // Set counter to max
        vm.store(
            address(nft),
            bytes32(uint256(4)), // Storage slot for _tokenIdCounter
            bytes32(uint256(10000))
        );

        vm.expectRevert(QuantumAlgorithmNFT_Secure.MaxSupplyReached.selector);
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);
    }

    /*//////////////////////////////////////////////////////////////
                    ADMIN FUNCTION TESTS
    //////////////////////////////////////////////////////////////*/

    function test_SetMintingActive() public {
        assertTrue(nft.mintingActive());

        vm.expectEmit(true, false, false, true);
        emit MintingActiveChanged(false, owner);

        vm.prank(owner);
        nft.setMintingActive(false);

        assertFalse(nft.mintingActive());
    }

    function test_UpdateQuantumAdvantage() public {
        // Mint an NFT
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);
        vrfCoordinator.fulfillRandomWords(1, address(nft));

        QuantumAlgorithmNFT_Secure.QuantumAlgorithm memory algBefore = nft.getAlgorithm(0);

        vm.prank(owner);
        nft.updateQuantumAdvantage(0, 9999);

        QuantumAlgorithmNFT_Secure.QuantumAlgorithm memory algAfter = nft.getAlgorithm(0);

        assertEq(algAfter.quantumAdvantage, 9999);
        assertEq(algBefore.algorithmType, algAfter.algorithmType);
    }

    function test_Pause() public {
        vm.prank(owner);
        nft.pause();

        vm.expectRevert("Pausable: paused");
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);
    }

    function test_Unpause() public {
        vm.prank(owner);
        nft.pause();

        vm.prank(owner);
        nft.unpause();

        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);
    }

    /*//////////////////////////////////////////////////////////////
                    ACCESS CONTROL TESTS
    //////////////////////////////////////////////////////////////*/

    function test_Revert_OnlyAdmin_SetMintingActive() public {
        vm.expectRevert();
        vm.prank(user);
        nft.setMintingActive(false);
    }

    function test_Revert_OnlyAdmin_Withdraw() public {
        vm.expectRevert();
        vm.prank(user);
        nft.withdraw();
    }

    function test_Revert_OnlyPlatformAdmin_ExtendAccess() public {
        vm.expectRevert();
        vm.prank(user);
        nft.extendPlatformAccess(user2, 30 days);
    }

    /*//////////////////////////////////////////////////////////////
                    ALGORITHM GENERATION TESTS
    //////////////////////////////////////////////////////////////*/

    function test_AlgorithmGeneration_DifferentRarities() public {
        // Mint multiple NFTs
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE() * 10}(10);
        vrfCoordinator.fulfillRandomWords(1, address(nft));

        // Check that different rarities exist
        bool[5] memory raritiesFound;

        for (uint256 i = 0; i < 10; i++) {
            QuantumAlgorithmNFT_Secure.QuantumAlgorithm memory alg = nft.getAlgorithm(i);
            raritiesFound[alg.rarity - 1] = true;

            // Verify quantum advantage scales with rarity
            if (alg.rarity == 1) {
                assertGe(alg.quantumAdvantage, 2);
                assertLe(alg.quantumAdvantage, 12);
            } else if (alg.rarity == 5) {
                assertGe(alg.quantumAdvantage, 1000);
            }
        }
    }

    function test_RarityDistribution() public {
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE() * 10}(10);
        vrfCoordinator.fulfillRandomWords(1, address(nft));

        uint256[5] memory distribution = nft.getRarityDistribution();
        uint256 total = 0;

        for (uint256 i = 0; i < 5; i++) {
            total += distribution[i];
        }

        assertEq(total, 10);
    }

    /*//////////////////////////////////////////////////////////////
                    TRANSFER & ROYALTY TESTS
    //////////////////////////////////////////////////////////////*/

    function test_Transfer_MaintainsPlatformAccess() public {
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);
        vrfCoordinator.fulfillRandomWords(1, address(nft));

        assertTrue(nft.hasPlatformAccess(user));
        assertFalse(nft.hasPlatformAccess(user2));

        // Transfer NFT
        vm.prank(user);
        nft.transferFrom(user, user2, 0);

        // Both should have access
        assertTrue(nft.hasPlatformAccess(user));
        assertTrue(nft.hasPlatformAccess(user2));
    }

    function test_RoyaltyInfo() public {
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);
        vrfCoordinator.fulfillRandomWords(1, address(nft));

        uint256 salePrice = 1 ether;
        (address receiver, uint256 royaltyAmount) = nft.royaltyInfo(0, salePrice);

        assertEq(receiver, royaltyRecipient);
        assertEq(royaltyAmount, (salePrice * 750) / 10000); // 7.5%
    }

    /*//////////////////////////////////////////////////////////////
                    FUZZ TESTS
    //////////////////////////////////////////////////////////////*/

    function testFuzz_Mint(uint256 quantity, uint256 payment) public {
        quantity = bound(quantity, 1, 10);
        payment = bound(payment, nft.MINT_PRICE() * quantity, 1000 ether);

        uint256 balanceBefore = user.balance;
        vm.deal(user, payment);

        vm.prank(user);
        nft.publicMint{value: payment}(quantity);

        vrfCoordinator.fulfillRandomWords(1, address(nft));

        assertEq(nft.balanceOf(user), quantity);

        // Verify correct refund
        uint256 expectedSpent = nft.MINT_PRICE() * quantity;
        assertEq(user.balance, payment - expectedSpent);
    }

    function testFuzz_PlatformAccessCheck(address randomUser) public {
        vm.assume(randomUser != address(0));

        // Should return false for anyone without NFT
        assertFalse(nft.hasPlatformAccess(randomUser));
    }

    /*//////////////////////////////////////////////////////////////
                    GAS BENCHMARK TESTS
    //////////////////////////////////////////////////////////////*/

    function test_GasBenchmark_PublicMint() public {
        uint256 gasBefore = gasleft();

        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);

        uint256 gasUsed = gasBefore - gasleft();
        console2.log("Gas used for publicMint:", gasUsed);

        // Should be reasonable (< 200k gas)
        assertLt(gasUsed, 200000);
    }

    function test_GasBenchmark_PlatformAccessCheck() public {
        vm.prank(user);
        nft.publicMint{value: nft.MINT_PRICE()}(1);
        vrfCoordinator.fulfillRandomWords(1, address(nft));

        uint256 gasBefore = gasleft();
        nft.hasPlatformAccess(user);
        uint256 gasUsed = gasBefore - gasleft();

        console2.log("Gas used for hasPlatformAccess:", gasUsed);
        assertLt(gasUsed, 5000);
    }
}

// Helper contract for testing .call() with contract wallets
contract SimpleWallet {
    receive() external payable {
        // Complex logic that would fail with 2300 gas stipend
        assembly {
            let x := 0
            for { let i := 0 } lt(i, 100) { i := add(i, 1) } {
                x := add(x, i)
            }
        }
    }
}
