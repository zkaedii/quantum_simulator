// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/utils/Strings.sol";

/**
 * @title QuantumAlgorithmsNFT
 * @dev NFT collection representing quantum computing algorithms with utility
 * Each NFT grants access to the Quantum Computing Platform
 */
contract QuantumAlgorithmsNFT is ERC721, ERC721URIStorage, Ownable, ReentrancyGuard {
    using Counters for Counters.Counter;
    using Strings for uint256;

    // Token counter
    Counters.Counter private _tokenIdCounter;

    // Collection parameters
    uint256 public constant MAX_SUPPLY = 10000;
    uint256 public constant MINT_PRICE = 0.08 ether; // ~$150 at current ETH prices
    uint256 public constant MAX_MINT_PER_TX = 10;
    uint256 public constant WHITELIST_MINT_PRICE = 0.06 ether; // 25% discount
    
    // Sale phases
    bool public whitelistSaleActive = false;
    bool public publicSaleActive = false;
    
    // Base URI for metadata
    string private _baseTokenURI;
    
    // Quantum platform access
    mapping(uint256 => bool) public platformAccess;
    mapping(uint256 => string) public quantumAlgorithm;
    mapping(uint256 => uint256) public quantumAdvantage;
    mapping(uint256 => string) public rarityLevel;
    
    // Whitelist for early access
    mapping(address => bool) public whitelist;
    mapping(address => uint256) public whitelistMinted;
    uint256 public constant WHITELIST_MINT_LIMIT = 5;
    
    // Royalty info
    address public royaltyReceiver;
    uint96 public royaltyFeeNumerator = 750; // 7.5%
    
    // Events
    event QuantumNFTMinted(uint256 indexed tokenId, address indexed owner, string algorithm, uint256 advantage);
    event PlatformAccessGranted(uint256 indexed tokenId, address indexed owner);
    event WhitelistStatusChanged(address indexed user, bool status);

    constructor(
        string memory name,
        string memory symbol,
        string memory baseTokenURI,
        address _royaltyReceiver
    ) ERC721(name, symbol) {
        _baseTokenURI = baseTokenURI;
        royaltyReceiver = _royaltyReceiver;
        
        // Start token IDs at 1
        _tokenIdCounter.increment();
    }

    /**
     * @dev Whitelist mint function - discounted price for early supporters
     */
    function whitelistMint(
        uint256 quantity,
        string[] calldata algorithms,
        uint256[] calldata advantages,
        string[] calldata rarities
    ) external payable nonReentrant {
        require(whitelistSaleActive, "Whitelist sale not active");
        require(whitelist[msg.sender], "Not whitelisted");
        require(quantity > 0 && quantity <= MAX_MINT_PER_TX, "Invalid quantity");
        require(whitelistMinted[msg.sender] + quantity <= WHITELIST_MINT_LIMIT, "Exceeds whitelist limit");
        require(totalSupply() + quantity <= MAX_SUPPLY, "Exceeds max supply");
        require(msg.value >= WHITELIST_MINT_PRICE * quantity, "Insufficient payment");
        require(algorithms.length == quantity, "Algorithm array length mismatch");
        require(advantages.length == quantity, "Advantage array length mismatch");
        require(rarities.length == quantity, "Rarity array length mismatch");

        whitelistMinted[msg.sender] += quantity;

        for (uint256 i = 0; i < quantity; i++) {
            uint256 tokenId = _tokenIdCounter.current();
            _tokenIdCounter.increment();

            _mint(msg.sender, tokenId);
            
            // Set quantum properties
            quantumAlgorithm[tokenId] = algorithms[i];
            quantumAdvantage[tokenId] = advantages[i];
            rarityLevel[tokenId] = rarities[i];
            platformAccess[tokenId] = true;

            emit QuantumNFTMinted(tokenId, msg.sender, algorithms[i], advantages[i]);
            emit PlatformAccessGranted(tokenId, msg.sender);
        }
    }

    /**
     * @dev Public mint function - standard pricing
     */
    function publicMint(
        uint256 quantity,
        string[] calldata algorithms,
        uint256[] calldata advantages,
        string[] calldata rarities
    ) external payable nonReentrant {
        require(publicSaleActive, "Public sale not active");
        require(quantity > 0 && quantity <= MAX_MINT_PER_TX, "Invalid quantity");
        require(totalSupply() + quantity <= MAX_SUPPLY, "Exceeds max supply");
        require(msg.value >= MINT_PRICE * quantity, "Insufficient payment");
        require(algorithms.length == quantity, "Algorithm array length mismatch");
        require(advantages.length == quantity, "Advantage array length mismatch");
        require(rarities.length == quantity, "Rarity array length mismatch");

        for (uint256 i = 0; i < quantity; i++) {
            uint256 tokenId = _tokenIdCounter.current();
            _tokenIdCounter.increment();

            _mint(msg.sender, tokenId);
            
            // Set quantum properties
            quantumAlgorithm[tokenId] = algorithms[i];
            quantumAdvantage[tokenId] = advantages[i];
            rarityLevel[tokenId] = rarities[i];
            platformAccess[tokenId] = true;

            emit QuantumNFTMinted(tokenId, msg.sender, algorithms[i], advantages[i]);
            emit PlatformAccessGranted(tokenId, msg.sender);
        }
    }

    /**
     * @dev Owner mint for team/marketing/partnerships
     */
    function ownerMint(
        address to,
        uint256 quantity,
        string[] calldata algorithms,
        uint256[] calldata advantages,
        string[] calldata rarities
    ) external onlyOwner {
        require(quantity > 0, "Invalid quantity");
        require(totalSupply() + quantity <= MAX_SUPPLY, "Exceeds max supply");
        require(algorithms.length == quantity, "Algorithm array length mismatch");
        require(advantages.length == quantity, "Advantage array length mismatch");
        require(rarities.length == quantity, "Rarity array length mismatch");

        for (uint256 i = 0; i < quantity; i++) {
            uint256 tokenId = _tokenIdCounter.current();
            _tokenIdCounter.increment();

            _mint(to, tokenId);
            
            // Set quantum properties
            quantumAlgorithm[tokenId] = algorithms[i];
            quantumAdvantage[tokenId] = advantages[i];
            rarityLevel[tokenId] = rarities[i];
            platformAccess[tokenId] = true;

            emit QuantumNFTMinted(tokenId, to, algorithms[i], advantages[i]);
            emit PlatformAccessGranted(tokenId, to);
        }
    }

    /**
     * @dev Add addresses to whitelist
     */
    function addToWhitelist(address[] calldata addresses) external onlyOwner {
        for (uint256 i = 0; i < addresses.length; i++) {
            whitelist[addresses[i]] = true;
            emit WhitelistStatusChanged(addresses[i], true);
        }
    }

    /**
     * @dev Remove addresses from whitelist
     */
    function removeFromWhitelist(address[] calldata addresses) external onlyOwner {
        for (uint256 i = 0; i < addresses.length; i++) {
            whitelist[addresses[i]] = false;
            emit WhitelistStatusChanged(addresses[i], false);
        }
    }

    /**
     * @dev Toggle whitelist sale status
     */
    function setWhitelistSaleActive(bool active) external onlyOwner {
        whitelistSaleActive = active;
    }

    /**
     * @dev Toggle public sale status
     */
    function setPublicSaleActive(bool active) external onlyOwner {
        publicSaleActive = active;
    }

    /**
     * @dev Set base URI for metadata
     */
    function setBaseURI(string calldata baseURI) external onlyOwner {
        _baseTokenURI = baseURI;
    }

    /**
     * @dev Check if address has platform access through NFT ownership
     */
    function hasPlatformAccess(address user) external view returns (bool) {
        uint256 balance = balanceOf(user);
        if (balance == 0) return false;
        
        // Check if any owned NFT has platform access
        for (uint256 i = 1; i <= totalSupply(); i++) {
            if (_exists(i) && ownerOf(i) == user && platformAccess[i]) {
                return true;
            }
        }
        return false;
    }

    /**
     * @dev Get quantum properties for a token
     */
    function getQuantumProperties(uint256 tokenId) external view returns (
        string memory algorithm,
        uint256 advantage,
        string memory rarity,
        bool access
    ) {
        require(_exists(tokenId), "Token does not exist");
        return (
            quantumAlgorithm[tokenId],
            quantumAdvantage[tokenId],
            rarityLevel[tokenId],
            platformAccess[tokenId]
        );
    }

    /**
     * @dev Get all tokens owned by an address with quantum properties
     */
    function getOwnerTokens(address owner) external view returns (
        uint256[] memory tokenIds,
        string[] memory algorithms,
        uint256[] memory advantages,
        string[] memory rarities
    ) {
        uint256 ownerBalance = balanceOf(owner);
        uint256[] memory ownedTokenIds = new uint256[](ownerBalance);
        string[] memory ownedAlgorithms = new string[](ownerBalance);
        uint256[] memory ownedAdvantages = new uint256[](ownerBalance);
        string[] memory ownedRarities = new string[](ownerBalance);
        
        uint256 currentIndex = 0;
        for (uint256 i = 1; i <= totalSupply() && currentIndex < ownerBalance; i++) {
            if (_exists(i) && ownerOf(i) == owner) {
                ownedTokenIds[currentIndex] = i;
                ownedAlgorithms[currentIndex] = quantumAlgorithm[i];
                ownedAdvantages[currentIndex] = quantumAdvantage[i];
                ownedRarities[currentIndex] = rarityLevel[i];
                currentIndex++;
            }
        }
        
        return (ownedTokenIds, ownedAlgorithms, ownedAdvantages, ownedRarities);
    }

    /**
     * @dev Get total supply
     */
    function totalSupply() public view returns (uint256) {
        return _tokenIdCounter.current() - 1;
    }

    /**
     * @dev Withdraw contract balance
     */
    function withdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No funds to withdraw");
        
        (bool success, ) = payable(owner()).call{value: balance}("");
        require(success, "Withdrawal failed");
    }

    /**
     * @dev Set royalty info
     */
    function setRoyaltyInfo(address receiver, uint96 feeNumerator) external onlyOwner {
        royaltyReceiver = receiver;
        royaltyFeeNumerator = feeNumerator;
    }

    // Override functions
    function _baseURI() internal view override returns (string memory) {
        return _baseTokenURI;
    }

    function tokenURI(uint256 tokenId) public view override(ERC721, ERC721URIStorage) returns (string memory) {
        return super.tokenURI(tokenId);
    }

    function _burn(uint256 tokenId) internal override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
    }

    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view override(ERC721) returns (bool) {
        return interfaceId == 0x2a55205a || super.supportsInterface(interfaceId); // ERC2981
    }

    /**
     * @dev See {IERC2981-royaltyInfo}.
     */
    function royaltyInfo(uint256 tokenId, uint256 salePrice) external view returns (address, uint256) {
        require(_exists(tokenId), "Token does not exist");
        uint256 royaltyAmount = (salePrice * royaltyFeeNumerator) / 10000;
        return (royaltyReceiver, royaltyAmount);
    }
} 