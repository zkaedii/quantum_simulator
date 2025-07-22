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
     * @dev Mint a quantum algorithm NFT with specified properties
     */
    function mintAlgorithm(
        address to,
        string calldata algorithmType,
        uint256 quantumAdvantage,
        string calldata complexityClass,
        uint8 rarity,
        uint256 qubitsRequired
    ) external onlyRole(MINTER_ROLE) whenNotPaused nonReentrant {
        require(to != address(0), "Cannot mint to zero address");
        require(_tokenIdCounter.current() < MAX_SUPPLY, "Max supply exceeded");
        require(rarity >= 1 && rarity <= 5, "Invalid rarity");
        require(quantumAdvantage > 0, "Quantum advantage must be positive");
        require(bytes(algorithmType).length > 0, "Algorithm type required");

        uint256 tokenId = _tokenIdCounter.current();
        _tokenIdCounter.increment();

        // Store algorithm data
        algorithms[tokenId] = QuantumAlgorithm({
            algorithmType: algorithmType,
            quantumAdvantage: quantumAdvantage,
            complexityClass: complexityClass,
            rarity: rarity,
            qubitsRequired: qubitsRequired,
            platformAccess: true,
            mintTimestamp: block.timestamp
        });

        // Update rarity supply
        raritySupply[rarity]++;

        // Grant 1 year platform access
        platformAccessExpiry[to] = block.timestamp + 365 days;

        _mint(to, tokenId);

        emit AlgorithmMinted(to, tokenId, algorithmType, quantumAdvantage);
        emit PlatformAccessGranted(to, platformAccessExpiry[to]);
    }

    /**
     * @dev Public minting function
     */
    function publicMint(
        string calldata algorithmType,
        uint256 quantumAdvantage,
        string calldata complexityClass,
        uint8 rarity,
        uint256 qubitsRequired
    ) external payable whenNotPaused nonReentrant {
        require(mintingActive, "Minting not active");
        require(msg.value >= MINT_PRICE, "Insufficient payment");
        require(_tokenIdCounter.current() < MAX_SUPPLY, "Max supply exceeded");

        // Refund excess payment
        if (msg.value > MINT_PRICE) {
            payable(msg.sender).transfer(msg.value - MINT_PRICE);
        }

        // Forward to mint function
        this.mintAlgorithm(
            msg.sender,
            algorithmType,
            quantumAdvantage,
            complexityClass,
            rarity,
            qubitsRequired
        );
    }

    /**
     * @dev Check if user has active platform access
     */
    function hasPlatformAccess(address user) external view returns (bool) {
        return platformAccessExpiry[user] > block.timestamp;
    }

    /**
     * @dev Update quantum advantage for an algorithm (for research purposes)
     */
    function updateQuantumAdvantage(
        uint256 tokenId, 
        uint256 newAdvantage
    ) external onlyRole(PLATFORM_ADMIN_ROLE) {
        require(_exists(tokenId), "Token does not exist");
        require(newAdvantage > 0, "Advantage must be positive");
        
        algorithms[tokenId].quantumAdvantage = newAdvantage;
        emit QuantumAdvantageUpdated(tokenId, newAdvantage);
    }

    /**
     * @dev Set minting active state
     */
    function setMintingActive(bool active) external onlyRole(DEFAULT_ADMIN_ROLE) {
        mintingActive = active;
    }

    /**
     * @dev Set platform contract address
     */
    function setPlatformContract(address _platformContract) external onlyRole(DEFAULT_ADMIN_ROLE) {
        platformContract = _platformContract;
    }

    /**
     * @dev Withdraw contract balance
     */
    function withdraw() external onlyRole(DEFAULT_ADMIN_ROLE) {
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
     * @dev Get total algorithms minted
     */
    function totalAlgorithms() external view returns (uint256) {
        return _tokenIdCounter.current();
    }

    /**
     * @dev Get algorithm data
     */
    function getAlgorithm(uint256 tokenId) external view returns (QuantumAlgorithm memory) {
        require(_exists(tokenId), "Token does not exist");
        return algorithms[tokenId];
    }

    /**
     * @dev EIP-2981 royalty standard
     */
    function royaltyInfo(uint256 tokenId, uint256 salePrice) 
        external 
        view 
        override 
        returns (address receiver, uint256 royaltyAmount) 
    {
        require(_exists(tokenId), "Token does not exist");
        return (royaltyRecipient, (salePrice * ROYALTY_BPS) / 10000);
    }

    // Required overrides
    function _beforeTokenTransfer(
        address from, 
        address to, 
        uint256 tokenId, 
        uint256 batchSize
    ) internal override(ERC721, ERC721Enumerable) whenNotPaused {
        super._beforeTokenTransfer(from, to, tokenId, batchSize);
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
        override(ERC721, ERC721Enumerable, ERC721URIStorage, AccessControl, IERC165) 
        returns (bool) 
    {
        return super.supportsInterface(interfaceId) || interfaceId == type(IERC2981).interfaceId;
    }
} 