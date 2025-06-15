# advanced_tokenizer.py
"""
Advanced tokenizer with subword tokenization and semantic mapping
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import re
import json
from torus_geometry import TorusPosition

class BPETokenizer:
    """Byte Pair Encoding tokenizer with torus position mapping"""
    
    def __init__(self, vocab_size: int = 5000, R: float = 3.0, r: float = 1.0):
        self.vocab_size = vocab_size
        self.R = R
        self.r = r
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1, 
            '<START>': 2,
            '<END>': 3,
            '<SEP>': 4,
            '<MASK>': 5
        }
        
        # Token IDs for easy access
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.start_token_id = 2
        self.eos_token_id = 3  # end of sequence
        self.sep_token_id = 4
        self.mask_token_id = 5
        
        # Initialize vocabulary with special tokens and single characters
        self.vocab = dict(self.special_tokens)
        self.positions = {}  # token -> TorusPosition
        
        # BPE specific
        self.merges = []  # List of merge rules
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Semantic categories for position mapping
        self.semantic_patterns = {
            # Linguistic patterns
            'prefix': (0, np.pi/4),              # Common prefixes
            'suffix': (np.pi/4, np.pi/2),        # Common suffixes
            'root': (np.pi/2, np.pi),            # Root words
            'compound': (np.pi, 3*np.pi/2),      # Compound parts
            'special': (3*np.pi/2, 2*np.pi),     # Special tokens
        }
        
        # Initialize positions for special tokens
        self._init_special_positions()
    
    def _bytes_to_unicode(self):
        """Map bytes to unicode strings to avoid control tokens"""
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def _init_special_positions(self):
        """Initialize positions for special tokens"""
        special_region = self.semantic_patterns['special']
        u_start, u_end = special_region
        
        for i, (token, idx) in enumerate(self.special_tokens.items()):
            # Distribute special tokens evenly in special region
            u = u_start + (u_end - u_start) * (i / len(self.special_tokens))
            v = np.pi  # Middle of minor circle
            self.positions[token] = TorusPosition(u, v, self.R, self.r)
    
    def train(self, texts: List[str], min_frequency: int = 2):
        """Train BPE tokenizer on corpus"""
        print("Training BPE tokenizer...")
        
        # Step 1: Pre-tokenize and get character vocabulary
        word_freqs = self._get_word_frequencies(texts)
        
        # Step 2: Initialize vocabulary with all characters
        alphabet = set()
        for word in word_freqs:
            alphabet.update(word)
        
        for char in alphabet:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        
        # Step 3: Learn BPE merges
        splits = {word: list(word) for word in word_freqs}
        
        while len(self.vocab) < self.vocab_size:
            # Find most frequent pair
            pair_freqs = self._compute_pair_frequencies(splits, word_freqs)
            
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            if pair_freqs[best_pair] < min_frequency:
                break
            
            # Merge the best pair
            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
            
            # Update splits
            splits = self._merge_pair(best_pair, splits)
            
            if len(self.vocab) % 100 == 0:
                print(f"Vocabulary size: {len(self.vocab)}")
        
        # Step 4: Assign positions to tokens based on linguistic properties
        self._assign_torus_positions()
        
        print(f"Training complete. Vocabulary size: {len(self.vocab)}")
    
    def _get_word_frequencies(self, texts: List[str]) -> Dict[str, int]:
        """Count word frequencies in corpus"""
        word_freqs = Counter()
        for text in texts:
            # Simple whitespace + punctuation tokenization
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            word_freqs.update(words)
        return dict(word_freqs)
    
    def _compute_pair_frequencies(self, splits: Dict[str, List[str]], 
                                 word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Compute frequencies of adjacent pairs"""
        pair_freqs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            split = splits.get(word, [])
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        
        return dict(pair_freqs)
    
    def _merge_pair(self, pair: Tuple[str, str], 
                   splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge a pair in all words"""
        new_splits = {}
        
        for word, split in splits.items():
            new_split = []
            i = 0
            
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                    new_split.append(split[i] + split[i + 1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            
            new_splits[word] = new_split
        
        return new_splits
    
    def _assign_torus_positions(self):
        """Assign torus positions to tokens based on linguistic properties"""
        for token in self.vocab:
            if token in self.special_tokens or token in self.positions:
                continue
            
            # Determine token type
            if len(token) == 1:
                # Single character - distribute around torus
                char_idx = ord(token[0]) if len(token) == 1 else 0
                u = (char_idx / 128) * 2 * np.pi  # Spread characters around major circle
                v = np.pi/2  # Lower on minor circle
            elif self._is_prefix(token):
                region = self.semantic_patterns['prefix']
                u = np.random.uniform(*region)
                v = np.random.uniform(0, np.pi)
            elif self._is_suffix(token):
                region = self.semantic_patterns['suffix']
                u = np.random.uniform(*region)
                v = np.random.uniform(np.pi, 2*np.pi)
            elif len(token) > 5:
                # Likely a root word or compound
                region = self.semantic_patterns['root'] if len(token) < 8 else self.semantic_patterns['compound']
                u = np.random.uniform(*region)
                v = np.random.uniform(0, 2*np.pi)
            else:
                # Default: random position
                u = np.random.uniform(0, 2*np.pi)
                v = np.random.uniform(0, 2*np.pi)
            
            # Add small noise to prevent overlap
            u += np.random.normal(0, 0.05)
            v += np.random.normal(0, 0.05)
            
            # Wrap to [0, 2π]
            u = u % (2 * np.pi)
            v = v % (2 * np.pi)
            
            self.positions[token] = TorusPosition(u, v, self.R, self.r)
    
    def _is_prefix(self, token: str) -> bool:
        """Check if token is likely a prefix"""
        common_prefixes = ['un', 're', 'pre', 'dis', 'over', 'under', 'mis', 'sub', 'inter', 'fore', 'de', 'in', 'im', 'il', 'ir']
        return any(token == prefix or token.startswith(prefix) for prefix in common_prefixes)
    
    def _is_suffix(self, token: str) -> bool:
        """Check if token is likely a suffix"""
        common_suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'ity', 'ment', 'ness', 'tion', 'sion', 'able', 'ible', 'ful', 'less', 'ize', 'ise']
        return any(token == suffix or token.endswith(suffix) for suffix in common_suffixes)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using learned BPE rules"""
        # Pre-tokenize into words
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        
        tokens = []
        for word in words:
            # Apply BPE splits
            word_tokens = self._bpe_tokenize_word(word)
            tokens.extend(word_tokens)
        
        return tokens
    
    def _bpe_tokenize_word(self, word: str) -> List[str]:
        """Apply BPE to a single word"""
        if word in self.vocab:
            return [word]
        
        # Start with character splits
        splits = list(word)
        
        # Apply merges in order
        for pair in self.merges:
            new_splits = []
            i = 0
            
            while i < len(splits):
                if i < len(splits) - 1 and (splits[i], splits[i + 1]) == pair:
                    new_splits.append(splits[i] + splits[i + 1])
                    i += 2
                else:
                    new_splits.append(splits[i])
                    i += 1
            
            splits = new_splits
        
        # Map to vocabulary
        result = []
        for token in splits:
            if token in self.vocab:
                result.append(token)
            else:
                # Handle unknown subwords
                result.append('<UNK>')
        
        return result
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token indices"""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
    
    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to text"""
        inv_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [inv_vocab.get(idx, '<UNK>') for idx in indices]
        
        # Simple concatenation with space handling
        text = ' '.join(tokens)
        # Clean up spaces around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        
        return text.strip()
    
    def get_positions(self, tokens: List[str]) -> List[TorusPosition]:
        """Get torus positions for tokens"""
        positions = []
        for token in tokens:
            if token in self.positions:
                positions.append(self.positions[token])
            else:
                # Unknown token - place randomly
                u = np.random.uniform(0, 2*np.pi)
                v = np.random.uniform(0, 2*np.pi)
                positions.append(TorusPosition(u, v, self.R, self.r))
        
        return positions
    
    def save(self, path: str):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'positions': {
                token: {'u': pos.u, 'v': pos.v}
                for token, pos in self.positions.items()
            },
            'config': {
                'vocab_size': self.vocab_size,
                'R': self.R,
                'r': self.r
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = [tuple(pair) for pair in data['merges']]
        self.vocab_size = data['config']['vocab_size']
        self.R = data['config']['R']
        self.r = data['config']['r']
        
        # Reconstruct positions
        self.positions = {}
        for token, pos_data in data['positions'].items():
            self.positions[token] = TorusPosition(
                pos_data['u'], pos_data['v'], self.R, self.r
            )


def demo_advanced_tokenizer():
    """Demo the advanced tokenizer"""
    print("🔤 Advanced Tokenizer Demo")
    print("="*50)
    
    # Sample corpus
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming artificial intelligence",
        "Natural language processing enables computers to understand text",
        "Deep learning models can generate human-like text",
        "Tokenization is the first step in text processing",
        "Subword tokenization handles unknown words better",
        "The transformer architecture revolutionized NLP",
        "Attention mechanisms allow models to focus on relevant parts"
    ] * 10  # Repeat for more training data
    
    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=500, R=3.0, r=1.0)
    tokenizer.train(corpus)
    
    # Test tokenization
    test_texts = [
        "The quick brown fox",
        "Unknown words like zobblefrog",
        "Tokenization handles prefixes and suffixes"
    ]
    
    print("\nTokenization examples:")
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        indices = tokenizer.encode(text)
        reconstructed = tokenizer.decode(indices)
        
        print(f"\nOriginal: {text}")
        print(f"Tokens: {tokens}")
        print(f"Indices: {indices}")
        print(f"Reconstructed: {reconstructed}")
    
    # Show position mapping
    print("\n\nSemantic position mapping:")
    example_tokens = ['the', 'ing', 'un', 'learn', 'transformer']
    for token in example_tokens:
        if token in tokenizer.positions:
            pos = tokenizer.positions[token]
            print(f"'{token}' -> (u={pos.u:.3f}, v={pos.v:.3f})")
    
    # Save tokenizer
    tokenizer.save('tokenizer.json')
    print("\n✅ Tokenizer saved to 'tokenizer.json'")


if __name__ == "__main__":
    demo_advanced_tokenizer()