// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/interfaces/IERC2981.sol";

/**
 * @title QuantumAlgorithmNFT
 * @dev NFT collection representing quantum algorithms with verified performance metrics
 * @notice Each NFT grants access to quantum computing platform and represents unique algorithm
 */
contract QuantumAlgorithmNFT is 
    ERC721, 
    ERC721Enumerable, 
    ERC721URIStorage, 
    Pausable, 
    AccessControl, 
    ReentrancyGuard, 
    IERC2981 
{
    using Counters for Counters.Counter;

    // Constants
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant PLATFORM_ADMIN_ROLE = keccak256("PLATFORM_ADMIN_ROLE");

    uint256 public constant MAX_SUPPLY = 10000;
    uint256 public constant ROYALTY_BPS = 750; // 7.5%
    uint256 public constant MINT_PRICE = 0.08 ether;

    // State Variables
    Counters.Counter private _tokenIdCounter;
    
    struct QuantumAlgorithm {
        string algorithmType;     // "Shor", "Grover", "VQE", etc.
        uint256 quantumAdvantage; // Performance multiplier vs classical
        string complexityClass;   // "BQP", "QMA", "P", etc.
        uint8 rarity;            // 1=Common, 2=Rare, 3=Epic, 4=Legendary, 5=Mythical
        uint256 qubitsRequired;  // Minimum qubits needed
        bool platformAccess;     // Access to quantum platform
        uint256 mintTimestamp;   // When minted
    }

    mapping(uint256 => QuantumAlgorithm) public algorithms;
    mapping(address => uint256) public platformAccessExpiry;
    mapping(uint8 => uint256) public raritySupply; // Track supply per rarity
    
    address public platformContract;
    address public royaltyRecipient;
    bool public mintingActive = false;
    
    // Events
    event AlgorithmMinted(
        address indexed to, 
        uint256 indexed tokenId, 
        string algorithmType, 
        uint256 quantumAdvantage
    );
    event PlatformAccessGranted(address indexed user, uint256 expiryTimestamp);
    event QuantumAdvantageUpdated(uint256 indexed tokenId, uint256 newAdvantage);

    constructor(
        address _royaltyRecipient
    ) ERC721("Quantum Algorithm Collection", "QUANTUM") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(PAUSER_ROLE, msg.sender);
        _grantRole(PLATFORM_ADMIN_ROLE, msg.sender);
        
        royaltyRecipient = _royaltyRecipient;
    }

    /**
     * @dev Mint quantum algorithm NFT with verified performance metrics
     * @param to Address to mint to
     * @param algorithmType Type of quantum algorithm
     * @param quantumAdvantage Performance advantage over classical
     * @param complexityClass Computational complexity class
     * @param rarity NFT rarity level (1-5)
     * @param qubitsRequired Minimum qubits needed
     * @param tokenURI Metadata URI for the NFT
     */
    function mintAlgorithm(
        address to,
        string memory algorithmType,
        uint256 quantumAdvantage,
        string memory complexityClass,
        uint8 rarity,
        uint256 qubitsRequired,
        string memory tokenURI
    ) external onlyRole(MINTER_ROLE) nonReentrant {
        require(mintingActive, "Minting not active");
        require(_tokenIdCounter.current() < MAX_SUPPLY, "Max supply reached");
        require(rarity >= 1 && rarity <= 5, "Invalid rarity");
        require(quantumAdvantage > 0, "Quantum advantage must be positive");
        require(bytes(algorithmType).length > 0, "Algorithm type required");

        uint256 tokenId = _tokenIdCounter.current();
        _tokenIdCounter.increment();

        // Create algorithm struct
        algorithms[tokenId] = QuantumAlgorithm({
            algorithmType: algorithmType,
            quantumAdvantage: quantumAdvantage,
            complexityClass: complexityClass,
            rarity: rarity,
            qubitsRequired: qubitsRequired,
            platformAccess: true,
            mintTimestamp: block.timestamp
        });

        // Update rarity tracking
        raritySupply[rarity]++;

        // Mint NFT
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, tokenURI);

        // Grant platform access (1 year)
        platformAccessExpiry[to] = block.timestamp + 365 days;

        emit AlgorithmMinted(to, tokenId, algorithmType, quantumAdvantage);
        emit PlatformAccessGranted(to, platformAccessExpiry[to]);
    }

    /**
     * @dev Public minting function with payment
     */
    function publicMint(uint256 quantity) external payable nonReentrant whenNotPaused {
        require(mintingActive, "Minting not active");
        require(quantity > 0 && quantity <= 10, "Invalid quantity");
        require(_tokenIdCounter.current() + quantity <= MAX_SUPPLY, "Would exceed max supply");
        require(msg.value >= MINT_PRICE * quantity, "Insufficient payment");

        for (uint256 i = 0; i < quantity; i++) {
            _generateRandomAlgorithm(msg.sender);
        }

        // Refund excess payment
        if (msg.value > MINT_PRICE * quantity) {
            payable(msg.sender).transfer(msg.value - (MINT_PRICE * quantity));
        }
    }

    /**
     * @dev Generate random algorithm with weighted rarity distribution
     */
    function _generateRandomAlgorithm(address to) internal {
        uint256 tokenId = _tokenIdCounter.current();
        _tokenIdCounter.increment();

        // Pseudo-random generation (use Chainlink VRF in production)
        uint256 randomSeed = uint256(keccak256(abi.encodePacked(
            block.timestamp, 
            block.difficulty, 
            msg.sender, 
            tokenId
        )));

        // Determine rarity (weighted distribution)
        uint8 rarity = _determineRarity(randomSeed);
        
        // Generate algorithm properties based on rarity
        (string memory algorithmType, 
         uint256 quantumAdvantage, 
         string memory complexityClass, 
         uint256 qubitsRequired,
         string memory tokenURI) = _generateAlgorithmProperties(rarity, randomSeed);

        algorithms[tokenId] = QuantumAlgorithm({
            algorithmType: algorithmType,
            quantumAdvantage: quantumAdvantage,
            complexityClass: complexityClass,
            rarity: rarity,
            qubitsRequired: qubitsRequired,
            platformAccess: true,
            mintTimestamp: block.timestamp
        });

        raritySupply[rarity]++;
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, tokenURI);

        // Grant platform access
        if (platformAccessExpiry[to] < block.timestamp + 365 days) {
            platformAccessExpiry[to] = block.timestamp + 365 days;
        }

        emit AlgorithmMinted(to, tokenId, algorithmType, quantumAdvantage);
    }

    /**
     * @dev Determine rarity based on weighted probabilities
     */
    function _determineRarity(uint256 randomSeed) internal pure returns (uint8) {
        uint256 roll = randomSeed % 10000;
        
        if (roll < 5000) return 1; // 50% Common
        if (roll < 7500) return 2; // 25% Rare  
        if (roll < 9000) return 3; // 15% Epic
        if (roll < 9900) return 4; // 9% Legendary
        return 5; // 1% Mythical
    }

    /**
     * @dev Generate algorithm properties based on rarity
     */
    function _generateAlgorithmProperties(uint8 rarity, uint256 seed) 
        internal 
        pure 
        returns (
            string memory algorithmType,
            uint256 quantumAdvantage,
            string memory complexityClass,
            uint256 qubitsRequired,
            string memory tokenURI
        ) 
    {
        string[5] memory types = ["Grover", "Shor", "VQE", "QAOA", "Quantum_ML"];
        string[3] memory classes = ["BQP", "QMA", "P"];
        
        algorithmType = types[seed % 5];
        complexityClass = classes[seed % 3];
        
        // Quantum advantage scales with rarity
        if (rarity == 1) {
            quantumAdvantage = 2 + (seed % 10); // 2-12x
            qubitsRequired = 5 + (seed % 15); // 5-20 qubits
        } else if (rarity == 2) {
            quantumAdvantage = 10 + (seed % 40); // 10-50x
            qubitsRequired = 15 + (seed % 25); // 15-40 qubits
        } else if (rarity == 3) {
            quantumAdvantage = 50 + (seed % 200); // 50-250x
            qubitsRequired = 30 + (seed % 50); // 30-80 qubits
        } else if (rarity == 4) {
            quantumAdvantage = 250 + (seed % 1000); // 250-1250x
            qubitsRequired = 60 + (seed % 100); // 60-160 qubits
        } else {
            quantumAdvantage = 1000 + (seed % 100000); // 1000-101000x
            qubitsRequired = 100 + (seed % 400); // 100-500 qubits
        }
        
        tokenURI = string(abi.encodePacked(
            "https://quantum-nft-api.com/metadata/",
            algorithmType,
            "_",
            Strings.toString(rarity)
        ));
    }

    /**
     * @dev Check if user has active platform access
     */
    function hasPlatformAccess(address user) external view returns (bool) {
        if (balanceOf(user) == 0) return false;
        return platformAccessExpiry[user] > block.timestamp;
    }

    /**
     * @dev Extend platform access for NFT holders
     */
    function extendPlatformAccess(address user, uint256 additionalTime) 
        external 
        onlyRole(PLATFORM_ADMIN_ROLE) 
    {
        require(balanceOf(user) > 0, "User must own NFT");
        platformAccessExpiry[user] += additionalTime;
        emit PlatformAccessGranted(user, platformAccessExpiry[user]);
    }

    /**
     * @dev Update quantum advantage (for verified improvements)
     */
    function updateQuantumAdvantage(uint256 tokenId, uint256 newAdvantage) 
        external 
        onlyRole(PLATFORM_ADMIN_ROLE) 
    {
        require(_exists(tokenId), "Token does not exist");
        require(newAdvantage > 0, "Advantage must be positive");
        
        algorithms[tokenId].quantumAdvantage = newAdvantage;
        emit QuantumAdvantageUpdated(tokenId, newAdvantage);
    }

    /**
     * @dev Get algorithm details for a token
     */
    function getAlgorithm(uint256 tokenId) external view returns (QuantumAlgorithm memory) {
        require(_exists(tokenId), "Token does not exist");
        return algorithms[tokenId];
    }

    /**
     * @dev Get rarity distribution
     */
    function getRarityDistribution() external view returns (uint256[5] memory) {
        return [
            raritySupply[1],
            raritySupply[2], 
            raritySupply[3],  // Fixed typo here
            raritySupply[4],
            raritySupply[5]
        ];
    }

    /**
     * @dev Toggle minting active state
     */
    function setMintingActive(bool _active) external onlyRole(DEFAULT_ADMIN_ROLE) {
        mintingActive = _active;
    }

    /**
     * @dev Withdraw contract balance
     */
    function withdraw() external onlyRole(DEFAULT_ADMIN_ROLE) nonReentrant {
        uint256 balance = address(this).balance;
        require(balance > 0, "No funds to withdraw");
        
        payable(royaltyRecipient).transfer(balance);
    }

    /**
     * @dev Emergency pause
     */
    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }

    /**
     * @dev Unpause
     */
    function unpause() external onlyRole(PAUSER_ROLE) {
        _unpause();
    }

    /**
     * @dev Royalty info for EIP-2981
     */
    function royaltyInfo(uint256 tokenId, uint256 salePrice) 
        external 
        view 
        override 
        returns (address receiver, uint256 royaltyAmount) 
    {
        receiver = royaltyRecipient;
        royaltyAmount = (salePrice * ROYALTY_BPS) / 10000;
    }

    // Required overrides
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 tokenId,
        uint256 batchSize
    ) internal override(ERC721, ERC721Enumerable) whenNotPaused {
        super._beforeTokenTransfer(from, to, tokenId, batchSize);
        
        // Transfer platform access if recipient doesn't have it
        if (to != address(0) && platformAccessExpiry[to] < block.timestamp + 365 days) {
            platformAccessExpiry[to] = block.timestamp + 365 days;
        }
    }

    function _burn(uint256 tokenId) internal override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
    }

    function tokenURI(uint256 tokenId)
        public
        view
        override(ERC721, ERC721URIStorage)
        returns (string memory)
    {
        return super.tokenURI(tokenId);
    }

    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC721, ERC721Enumerable, AccessControl, IERC165)
        returns (bool)
    {
        return 
            interfaceId == type(IERC2981).interfaceId ||
            super.supportsInterface(interfaceId);
    }
} 