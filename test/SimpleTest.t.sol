// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/SimpleTest.sol";

contract SimpleTestTest is Test {
    SimpleTest public simpleTest;

    function setUp() public {
        simpleTest = new SimpleTest();
    }

    function testName() public {
        assertEq(simpleTest.getName(), "Quantum NFT Test");
    }

    function testTotalSupply() public {
        assertEq(simpleTest.getTotalSupply(), 10000);
    }

    function testPublicVariables() public {
        assertEq(simpleTest.name(), "Quantum NFT Test");
        assertEq(simpleTest.totalSupply(), 10000);
    }
} 