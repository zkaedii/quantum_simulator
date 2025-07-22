// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Test} from "forge-std/Test.sol";
import {QuantumAlgorithmNFT} from "../src/QuantumAlgorithmNFT.sol";

/**
 * @title Quantum NFT Invariant Handler
 * @dev Advanced handler for invariant testing with quantum-specific properties
 * @author FoundryOps Advanced Testing
 */
contract QuantumNFTInvariantHandler is Test {
    QuantumAlgorithmNFT public nft;
    
    // State tracking
    uint256 public totalMinted;
    uint256 public totalWithdrawn;
    mapping(address => uint256) public userMintCount;
    mapping(uint8 => uint256) public rarityMinted;
    
    // Ghost variables for invariant tracking
    uint256 public ghost_sumOfQuantumAdvantages;
    uint256 public ghost_totalPlatformAccessUsers;
    
    // Actors for testing
    address[] public actors;
    address public currentActor;
    
    modifier useActor(uint256 actorIndexSeed) {
        currentActor = actors[bound(actorIndexSeed, 0, actors.length - 1)];
        vm.startPrank(currentActor);
        _;
        vm.stopPrank();
    }
    
    constructor(QuantumAlgorithmNFT _nft) {
        nft = _nft;
        
        // Set up actors
        actors.push(makeAddr("alice"));
        actors.push(makeAddr("bob"));
        actors.push(makeAddr("charlie"));
        actors.push(makeAddr("diana"));
        actors.push(makeAddr("eve"));
        
        // Fund actors
        for (uint256 i = 0; i < actors.length; i++) {
            vm.deal(actors[i], 100 ether);
        }
    }
    
    /*//////////////////////////////////////////////////////////////
                              ACTIONS
    //////////////////////////////////////////////////////////////*/
    
    function publicMint(uint256 actorSeed, uint256 quantity) external useActor(actorSeed) {
        quantity = bound(quantity, 1, 10);
        
        if (!nft.mintingActive()) return;
        if (nft.totalSupply() + quantity > nft.MAX_SUPPLY()) return;
        
        uint256 cost = nft.MINT_PRICE() * quantity;
        if (currentActor.balance < cost) return;
        
        uint256 balanceBefore = nft.balanceOf(currentActor);
        
        try nft.publicMint{value: cost}(quantity) {
            uint256 balanceAfter = nft.balanceOf(currentActor);
            uint256 actualMinted = balanceAfter - balanceBefore;
            
            totalMinted += actualMinted;
            userMintCount[currentActor] += actualMinted;
            
            // Update ghost variables
            for (uint256 i = 0; i < actualMinted; i++) {
                uint256 tokenId = nft.totalSupply() - actualMinted + i;
                QuantumAlgorithmNFT.QuantumAlgorithm memory algorithm = nft.getAlgorithm(tokenId);
                ghost_sumOfQuantumAdvantages += algorithm.quantumAdvantage;
                rarityMinted[algorithm.rarity]++;
            }
            
            if (nft.hasPlatformAccess(currentActor)) {
                ghost_totalPlatformAccessUsers++;
            }
        } catch {
            // Minting failed - that's ok for invariant testing
        }
    }
    
    function transferNFT(uint256 actorSeed, uint256 tokenIdSeed, uint256 recipientSeed) 
        external 
        useActor(actorSeed) 
    {
        if (nft.balanceOf(currentActor) == 0) return;
        
        uint256 tokenId = bound(tokenIdSeed, 0, nft.totalSupply() - 1);
        if (nft.ownerOf(tokenId) != currentActor) return;
        
        address recipient = actors[bound(recipientSeed, 0, actors.length - 1)];
        if (recipient == currentActor) return;
        
        bool recipientHadAccess = nft.hasPlatformAccess(recipient);
        
        try nft.transferFrom(currentActor, recipient, tokenId) {
            // Update platform access tracking
            if (!recipientHadAccess && nft.hasPlatformAccess(recipient)) {
                ghost_totalPlatformAccessUsers++;
            }
        } catch {
            // Transfer failed - that's ok
        }
    }
    
    function timeWarp(uint256 timeDelta) external {
        timeDelta = bound(timeDelta, 1 hours, 400 days);
        vm.warp(block.timestamp + timeDelta);
        
        // Recount platform access users after time change
        uint256 activeUsers = 0;
        for (uint256 i = 0; i < actors.length; i++) {
            if (nft.hasPlatformAccess(actors[i])) {
                activeUsers++;
            }
        }
        ghost_totalPlatformAccessUsers = activeUsers;
    }
    
    /*//////////////////////////////////////////////////////////////
                             INVARIANTS
    //////////////////////////////////////////////////////////////*/
    
    /// @dev Total supply should never exceed MAX_SUPPLY
    function invariant_A_totalSupplyBounded() external {
        assert(nft.totalSupply() <= nft.MAX_SUPPLY());
    }
    
    /// @dev Every existing NFT should have positive quantum advantage
    function invariant_B_quantumAdvantagePositive() external {
        uint256 totalSupply = nft.totalSupply();
        for (uint256 i = 0; i < totalSupply; i++) {
            QuantumAlgorithmNFT.QuantumAlgorithm memory algorithm = nft.getAlgorithm(i);
            assert(algorithm.quantumAdvantage > 0);
        }
    }
    
    /// @dev Rarity should always be between 1 and 5
    function invariant_C_rarityInBounds() external {
        uint256 totalSupply = nft.totalSupply();
        for (uint256 i = 0; i < totalSupply; i++) {
            QuantumAlgorithmNFT.QuantumAlgorithm memory algorithm = nft.getAlgorithm(i);
            assert(algorithm.rarity >= 1 && algorithm.rarity <= 5);
        }
    }
    
    /// @dev NFT owners should have platform access or expired access
    function invariant_D_ownershipImpliesAccess() external {
        for (uint256 i = 0; i < actors.length; i++) {
            address actor = actors[i];
            if (nft.balanceOf(actor) > 0) {
                bool hasAccess = nft.hasPlatformAccess(actor);
                bool accessExpired = nft.platformAccessExpiry(actor) <= block.timestamp;
                assert(hasAccess || accessExpired);
            }
        }
    }
    
    /// @dev Total minted should equal current supply
    function invariant_E_mintedEqualsSupply() external {
        assert(totalMinted == nft.totalSupply());
    }
    
    /// @dev Sum of quantum advantages should be consistent
    function invariant_F_quantumAdvantageSum() external {
        uint256 actualSum = 0;
        uint256 totalSupply = nft.totalSupply();
        
        for (uint256 i = 0; i < totalSupply; i++) {
            QuantumAlgorithmNFT.QuantumAlgorithm memory algorithm = nft.getAlgorithm(i);
            actualSum += algorithm.quantumAdvantage;
        }
        
        assert(actualSum == ghost_sumOfQuantumAdvantages);
    }
    
    /// @dev Contract balance should reflect received payments
    function invariant_G_contractBalance() external {
        uint256 expectedBalance = totalMinted * nft.MINT_PRICE() - totalWithdrawn;
        assert(address(nft).balance == expectedBalance);
    }
    
    /// @dev Platform access timestamps should be in the future for active users
    function invariant_H_activeAccessInFuture() external {
        for (uint256 i = 0; i < actors.length; i++) {
            address actor = actors[i];
            if (nft.hasPlatformAccess(actor)) {
                assert(nft.platformAccessExpiry(actor) > block.timestamp);
            }
        }
    }
} 