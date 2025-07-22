// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract SimpleTest {
    string public name = "Quantum NFT Test";
    uint256 public totalSupply = 10000;
    
    function getName() public view returns (string memory) {
        return name;
    }
    
    function getTotalSupply() public view returns (uint256) {
        return totalSupply;
    }
} 