#!/usr/bin/env python3
"""
Security Audit Verification Script
Validates the findings from SECURITY_AUDIT_REPORT.md
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Tuple

class AuditVerifier:
    def __init__(self):
        self.findings = []
        self.contract_dir = Path("/home/user/quantum_simulator")

    def verify_all(self):
        """Run all verification checks"""
        print("=" * 70)
        print("SECURITY AUDIT VERIFICATION")
        print("=" * 70)
        print()

        # Critical vulnerabilities
        self.verify_weak_randomness()
        self.verify_dos_vulnerability()

        # High severity
        self.verify_block_difficulty_usage()
        self.verify_unsafe_transfer()
        self.verify_missing_refund()
        self.verify_unbounded_arrays()

        # Medium severity
        self.verify_deprecated_counters()
        self.verify_test_coverage()
        self.verify_missing_events()

        # Summary
        self.print_summary()

    def verify_weak_randomness(self):
        """CRITICAL-01: Verify weak randomness vulnerability"""
        print("🔴 CRITICAL-01: Weak On-Chain Randomness")
        print("-" * 70)

        file_path = self.contract_dir / "QuantumAlgorithmNFT_Production.sol"
        with open(file_path, 'r') as f:
            content = f.read()

        # Check for vulnerable randomness pattern
        vulnerable_patterns = [
            r'block\.timestamp',
            r'block\.difficulty',
            r'block\.prevrandao',
            r'keccak256\(abi\.encodePacked\('
        ]

        found_issues = []
        for pattern in vulnerable_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                found_issues.append((pattern, line_num))

        if found_issues:
            print("❌ CONFIRMED: Weak randomness detected")
            for pattern, line_num in found_issues:
                print(f"   Line {line_num}: {pattern}")

            # Check if Chainlink VRF is used
            if 'VRFConsumerBase' not in content and 'VRFCoordinator' not in content:
                print("   ⚠️  Chainlink VRF NOT implemented")
            else:
                print("   ✅ Chainlink VRF found")
        else:
            print("✅ No weak randomness patterns found")

        print()
        self.findings.append(("CRITICAL-01", len(found_issues) > 0))

    def verify_dos_vulnerability(self):
        """CRITICAL-02: Verify DoS vulnerability in loops"""
        print("🔴 CRITICAL-02: Denial of Service - Unbounded Loop")
        print("-" * 70)

        file_path = self.contract_dir / "nft_smart_contract.sol"
        with open(file_path, 'r') as f:
            lines = f.readlines()
            content = ''.join(lines)

        # Find hasPlatformAccess function
        function_pattern = r'function hasPlatformAccess.*?\{(.*?)\n    \}'
        matches = re.finditer(function_pattern, content, re.DOTALL)

        found_vulnerable_loop = False
        for match in matches:
            function_body = match.group(1)
            # Check for loop through totalSupply
            if 'totalSupply()' in function_body and 'for' in function_body:
                start_line = content[:match.start()].count('\n') + 1
                print(f"❌ CONFIRMED: Unbounded loop found at line ~{start_line}")
                print(f"   Function loops through totalSupply() which can reach 10,000")
                print(f"   This will cause out-of-gas errors as collection grows")
                found_vulnerable_loop = True

        # Check getOwnerTokens too
        if 'getOwnerTokens' in content:
            if re.search(r'for.*totalSupply\(\)', content):
                print("❌ CONFIRMED: Similar issue in getOwnerTokens()")
                found_vulnerable_loop = True

        if not found_vulnerable_loop:
            print("✅ No unbounded loops found")

        print()
        self.findings.append(("CRITICAL-02", found_vulnerable_loop))

    def verify_block_difficulty_usage(self):
        """HIGH-01: Verify deprecated block.difficulty usage"""
        print("🟠 HIGH-01: Deprecated block.difficulty Usage")
        print("-" * 70)

        file_path = self.contract_dir / "QuantumAlgorithmNFT_Production.sol"
        with open(file_path, 'r') as f:
            content = f.read()

        matches = list(re.finditer(r'block\.difficulty', content))

        if matches:
            print(f"❌ CONFIRMED: block.difficulty used {len(matches)} time(s)")
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                print(f"   Line {line_num}: block.difficulty (deprecated post-Merge)")
            print("   ⚠️  Should use block.prevrandao or Chainlink VRF")
        else:
            print("✅ No block.difficulty usage found")

        print()
        self.findings.append(("HIGH-01", len(matches) > 0))

    def verify_unsafe_transfer(self):
        """HIGH-02: Verify unsafe .transfer() usage"""
        print("🟠 HIGH-02: Unsafe .transfer() for ETH Transfers")
        print("-" * 70)

        file_path = self.contract_dir / "QuantumAlgorithmNFT_Production.sol"
        with open(file_path, 'r') as f:
            content = f.read()

        # Find .transfer( usage
        matches = list(re.finditer(r'\.transfer\(', content))

        if matches:
            print(f"❌ CONFIRMED: .transfer() used {len(matches)} time(s)")
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                # Get context
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 50)
                context = content[start:end].strip()
                print(f"   Line {line_num}: {context[:80]}...")
            print("   ⚠️  2300 gas limit can cause failures with smart contracts")
            print("   ⚠️  Should use .call{value: amount}() instead")
        else:
            print("✅ No .transfer() usage found")

        print()
        self.findings.append(("HIGH-02", len(matches) > 0))

    def verify_missing_refund(self):
        """HIGH-03: Verify missing refund mechanism"""
        print("🟠 HIGH-03: Missing Refund Mechanism")
        print("-" * 70)

        file_path = self.contract_dir / "nft_smart_contract.sol"
        with open(file_path, 'r') as f:
            content = f.read()

        # Find mint functions
        mint_functions = re.finditer(
            r'function (whitelistMint|publicMint).*?\{(.*?)(?=\n    function|\n\})',
            content,
            re.DOTALL
        )

        missing_refund = []
        for match in mint_functions:
            func_name = match.group(1)
            func_body = match.group(2)

            # Check if function accepts payment but doesn't refund
            has_payment_check = 'msg.value' in func_body and 'require' in func_body
            has_refund = 'transfer' in func_body.lower() or 'call{value' in func_body

            if has_payment_check and not has_refund:
                line_num = content[:match.start()].count('\n') + 1
                missing_refund.append((func_name, line_num))

        if missing_refund:
            print(f"❌ CONFIRMED: {len(missing_refund)} function(s) missing refund logic")
            for func_name, line_num in missing_refund:
                print(f"   Line {line_num}: {func_name}() accepts payment but doesn't refund excess")
            print("   ⚠️  Users overpaying will lose their excess ETH")
        else:
            print("✅ All payment functions have refund logic")

        print()
        self.findings.append(("HIGH-03", len(missing_refund) > 0))

    def verify_unbounded_arrays(self):
        """HIGH-04: Verify unbounded array parameters"""
        print("🟠 HIGH-04: Unbounded Array Input Vulnerability")
        print("-" * 70)

        file_path = self.contract_dir / "nft_smart_contract.sol"
        with open(file_path, 'r') as f:
            content = f.read()

        # Find functions with array parameters
        array_params = re.finditer(
            r'function \w+\([^)]*\[\] (calldata|memory)[^)]*\)',
            content
        )

        vulnerable_functions = []
        for match in array_params:
            line_num = content[:match.start()].count('\n') + 1
            # Check if there's explicit array length validation
            func_start = match.start()
            func_end = content.find('\n    }', func_start)
            func_body = content[func_start:func_end]

            # Look for length validation
            has_length_check = '.length <=' in func_body or '.length >' in func_body

            if not has_length_check:
                vulnerable_functions.append((match.group(0), line_num))

        if vulnerable_functions:
            print(f"❌ CONFIRMED: {len(vulnerable_functions)} function(s) with unbounded arrays")
            for func_sig, line_num in vulnerable_functions[:3]:  # Show first 3
                print(f"   Line {line_num}: {func_sig[:60]}...")
            print("   ⚠️  Attackers can send huge arrays causing gas exhaustion")
        else:
            print("✅ All array parameters have length validation")

        print()
        self.findings.append(("HIGH-04", len(vulnerable_functions) > 0))

    def verify_deprecated_counters(self):
        """MEDIUM-01: Verify deprecated Counters usage"""
        print("🟡 MEDIUM-01: Deprecated OpenZeppelin Counters Library")
        print("-" * 70)

        file_path = self.contract_dir / "QuantumAlgorithmNFT_Production.sol"
        with open(file_path, 'r') as f:
            content = f.read()

        uses_counters = 'import "@openzeppelin/contracts/utils/Counters.sol"' in content
        uses_counters_lib = 'using Counters for Counters.Counter' in content

        if uses_counters or uses_counters_lib:
            print("❌ CONFIRMED: Deprecated Counters library in use")
            if uses_counters:
                print("   Import found: @openzeppelin/contracts/utils/Counters.sol")
            if uses_counters_lib:
                print("   Using statement found: using Counters for Counters.Counter")
            print("   ⚠️  Counters removed in OpenZeppelin v5.0")
            print("   ⚠️  Should use simple uint256 counter instead")
        else:
            print("✅ No deprecated Counters library usage")

        print()
        self.findings.append(("MEDIUM-01", uses_counters or uses_counters_lib))

    def verify_test_coverage(self):
        """MEDIUM-02: Verify test coverage"""
        print("🟡 MEDIUM-02: Test Coverage Analysis")
        print("-" * 70)

        test_dir = self.contract_dir / "test"
        if not test_dir.exists():
            print("❌ CONFIRMED: No test directory found")
            self.findings.append(("MEDIUM-02", True))
            print()
            return

        test_files = list(test_dir.glob("*.sol"))

        if not test_files:
            print("❌ CONFIRMED: No test files found")
        else:
            print(f"ℹ️  Found {len(test_files)} test file(s):")
            for test_file in test_files:
                size = test_file.stat().st_size
                print(f"   - {test_file.name} ({size} bytes)")

            # Check SimpleTest.t.sol
            simple_test = test_dir / "SimpleTest.t.sol"
            if simple_test.exists():
                with open(simple_test, 'r') as f:
                    content = f.read()
                    test_count = len(re.findall(r'function test\w+', content))
                    print(f"\n   SimpleTest.t.sol has {test_count} test functions")

                    if test_count < 5:
                        print("   ❌ MINIMAL: Only basic tests for dummy contract")
                        print("   ⚠️  No tests for production NFT contracts")
                        print("   ⚠️  No tests for critical functions (minting, access control)")
                        insufficient = True
                    else:
                        print("   ✅ Reasonable test coverage")
                        insufficient = False

        # Check if production contracts have corresponding tests
        prod_contracts = ["QuantumAlgorithmNFT_Production.sol", "nft_smart_contract.sol"]
        tested_contracts = [f.stem.replace(".t", "") for f in test_files]

        untested = [c for c in prod_contracts if c.replace(".sol", "") not in tested_contracts]

        if untested:
            print(f"\n   ❌ CONFIRMED: {len(untested)} production contract(s) without tests:")
            for contract in untested:
                print(f"      - {contract}")
            insufficient = True

        print()
        self.findings.append(("MEDIUM-02", insufficient if 'insufficient' in locals() else True))

    def verify_missing_events(self):
        """LOW-01: Verify missing events"""
        print("🔵 LOW-01: Missing Events for State Changes")
        print("-" * 70)

        file_path = self.contract_dir / "QuantumAlgorithmNFT_Production.sol"
        with open(file_path, 'r') as f:
            content = f.read()

        # Find setMintingActive function
        if 'function setMintingActive' in content:
            func_match = re.search(
                r'function setMintingActive.*?\{(.*?)\n    \}',
                content,
                re.DOTALL
            )
            if func_match:
                func_body = func_match.group(1)
                if 'emit' not in func_body:
                    line_num = content[:func_match.start()].count('\n') + 1
                    print(f"❌ CONFIRMED: setMintingActive() missing event emission")
                    print(f"   Line {line_num}: No event emitted for state change")
                else:
                    print("✅ setMintingActive() emits event")

        # Check other setter functions
        setter_pattern = r'function set\w+.*?\{(.*?)(?=\n    function|\n\})'
        setters = list(re.finditer(setter_pattern, content, re.DOTALL))

        missing_events = []
        for setter in setters:
            func_body = setter.group(1)
            if 'emit' not in func_body:
                line_num = content[:setter.start()].count('\n') + 1
                func_name = re.search(r'function (\w+)', setter.group(0)).group(1)
                missing_events.append((func_name, line_num))

        if missing_events:
            print(f"\n   Found {len(missing_events)} setter(s) without events:")
            for func_name, line_num in missing_events[:5]:
                print(f"      Line {line_num}: {func_name}()")

        print()
        self.findings.append(("LOW-01", len(missing_events) > 0))

    def print_summary(self):
        """Print verification summary"""
        print("=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)
        print()

        confirmed_count = sum(1 for _, confirmed in self.findings if confirmed)
        total_count = len(self.findings)

        print(f"Total Checks: {total_count}")
        print(f"Confirmed Vulnerabilities: {confirmed_count}")
        print(f"Verification Rate: {(confirmed_count/total_count*100):.1f}%")
        print()

        # Group by severity
        critical = [f for f in self.findings if f[0].startswith("CRITICAL") and f[1]]
        high = [f for f in self.findings if f[0].startswith("HIGH") and f[1]]
        medium = [f for f in self.findings if f[0].startswith("MEDIUM") and f[1]]
        low = [f for f in self.findings if f[0].startswith("LOW") and f[1]]

        print("Confirmed by Severity:")
        print(f"  🔴 CRITICAL: {len(critical)}")
        print(f"  🟠 HIGH: {len(high)}")
        print(f"  🟡 MEDIUM: {len(medium)}")
        print(f"  🔵 LOW: {len(low)}")
        print()

        if critical:
            print("⚠️  CRITICAL VULNERABILITIES CONFIRMED - DO NOT DEPLOY TO MAINNET")
        elif high:
            print("⚠️  HIGH SEVERITY ISSUES CONFIRMED - ADDRESS BEFORE PRODUCTION")
        elif medium:
            print("⚠️  MEDIUM SEVERITY ISSUES CONFIRMED - RECOMMENDED TO FIX")
        else:
            print("✅ No critical or high severity issues confirmed")

        print()
        print("=" * 70)
        print("Full audit report: SECURITY_AUDIT_REPORT.md")
        print("=" * 70)

if __name__ == "__main__":
    verifier = AuditVerifier()
    verifier.verify_all()
