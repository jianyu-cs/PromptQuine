from typing import List
from fastchat.model import get_conversation_template


def create_tabu_list(
        tokenizer,
        prompt_tokens: List[str],
        llm: str = None,
        in_context_jailbreak: bool = False,
    ):
    """
    Create a list of token positions that should not be modified (protects template placeholders)
    
    Args:
        tokenizer: The tokenizer instance
        prompt_tokens: List of tokenized prompt tokens
        llm: Language model name (required for in_context_jailbreak)
        in_context_jailbreak: Whether to protect conversation template tokens
    
    Returns:
        list: List of indices that should not be modified
    """
    tabu_list = []
    
    # Step 1: Protect standard placeholders
    tabu_list.extend(_protect_placeholders(tokenizer, prompt_tokens))
    
    # Step 2: Protect conversation template (for in-context attacks)
    if in_context_jailbreak:
        tabu_list.extend(_protect_conversation_template(tokenizer, prompt_tokens, llm))
    
    # Remove duplicates and sort
    tabu_list = sorted(set(tabu_list))
    
    return tabu_list


def _protect_placeholders(tokenizer, prompt_tokens: List[str]) -> List[int]:
    """
    Protect standard template placeholders from modification
    
    Args:
        tokenizer: The tokenizer instance
        prompt_tokens: List of tokenized prompt tokens
    
    Returns:
        list: Indices of protected placeholder tokens
    """
    PLACEHOLDERS = [
        '{sentence_1}',
        '{sentence_2}',
        '{question_1}',
        '{option_1}',
        '{option_2}',
        '{mask_token}'
    ]
    BRACE_TOKENS = [':{', '{', 'Ġ{', ' {', '▁{']
    
    protected_indices = []
    i = 0
    
    while i < len(prompt_tokens):
        # Check if current token could start a placeholder
        if prompt_tokens[i] in BRACE_TOKENS:
            # Try to match each placeholder
            for placeholder in PLACEHOLDERS:
                match_indices = _try_match_placeholder(
                    tokenizer, 
                    prompt_tokens, 
                    i, 
                    placeholder
                )
                if match_indices:
                    protected_indices.extend(match_indices)
                    break
        
        i += 1
    
    return protected_indices


def _try_match_placeholder(
        tokenizer, 
        prompt_tokens: List[str], 
        start_idx: int, 
        placeholder: str
    ) -> List[int]:
    """
    Try to match a placeholder starting at a given position
    
    Args:
        tokenizer: The tokenizer instance
        prompt_tokens: List of tokenized prompt tokens
        start_idx: Starting index to check for match
        placeholder: Placeholder string to match
    
    Returns:
        list: List of matched indices, or empty list if no match
    """
    # Tokenize placeholder with space prefix (common in transformers)
    placeholder_tokens = tokenizer.tokenize(f" {placeholder}")
    placeholder_len = len(placeholder_tokens)
    
    # Check if there's enough space for the placeholder
    if start_idx + placeholder_len > len(prompt_tokens):
        return []
    
    # Extract candidate tokens and join them
    candidate_tokens = prompt_tokens[start_idx:start_idx + placeholder_len]
    candidate_str = ''.join(candidate_tokens)
    
    # Check if placeholder is in the candidate string
    if placeholder in candidate_str:
        return list(range(start_idx, start_idx + placeholder_len))
    
    return []


def _protect_conversation_template(
        tokenizer, 
        prompt_tokens: List[str], 
        llm: str
    ) -> List[int]:
    """
    Protect conversation template tokens (system prompt, user tag, assistant tag)
    for in-context jailbreak attacks
    
    Args:
        tokenizer: The tokenizer instance
        prompt_tokens: List of tokenized prompt tokens
        llm: Language model name
    
    Returns:
        list: Indices of protected conversation template tokens
    """
    assert llm in ["vicuna_1.5", "llama-2"], \
        "We only support vicuna_1.5 and llama-2 for in-context attacks."
    
    # Get conversation template
    conv_template = get_conversation_template(llm)
    system_prompt = conv_template.get_prompt()
    user_tag = conv_template.roles[0]
    assistant_tag = conv_template.roles[1]
    
    # Collect all template strings to protect
    template_strings = [system_prompt, user_tag, assistant_tag]
    
    protected_indices = []
    
    # Protect each template string
    for template_str in template_strings:
        if template_str:  # Skip empty strings
            indices = _find_and_protect_string(
                tokenizer, 
                prompt_tokens, 
                template_str
            )
            protected_indices.extend(indices)
    
    return protected_indices


def _find_and_protect_string(
        tokenizer, 
        prompt_tokens: List[str], 
        target_string: str
    ) -> List[int]:
    """
    Find all occurrences of a target string in the prompt and return their indices
    
    Args:
        tokenizer: The tokenizer instance
        prompt_tokens: List of tokenized prompt tokens
        target_string: String to find and protect
    
    Returns:
        list: All indices where the target string occurs
    """
    protected_indices = []
    
    # Tokenize target string (try multiple variants)
    target_variants = [
        tokenizer.tokenize(target_string),
        tokenizer.tokenize(f" {target_string}"),
        tokenizer.tokenize(f"\n{target_string}"),
    ]
    
    # Try each variant
    for target_tokens in target_variants:
        if not target_tokens:
            continue
            
        target_len = len(target_tokens)
        
        # Sliding window to find matches
        for i in range(len(prompt_tokens) - target_len + 1):
            # Skip already protected positions
            if any(idx in protected_indices for idx in range(i, i + target_len)):
                continue
            
            # Check for exact match
            if _tokens_match(prompt_tokens[i:i + target_len], target_tokens):
                protected_indices.extend(range(i, i + target_len))
            # Check for fuzzy match (joining tokens)
            elif _fuzzy_match(prompt_tokens[i:i + target_len], target_string):
                protected_indices.extend(range(i, i + target_len))
    
    return protected_indices


def _tokens_match(tokens1: List[str], tokens2: List[str]) -> bool:
    """
    Check if two token lists match exactly
    
    Args:
        tokens1: First token list
        tokens2: Second token list
    
    Returns:
        bool: True if tokens match
    """
    if len(tokens1) != len(tokens2):
        return False
    
    return all(t1 == t2 for t1, t2 in zip(tokens1, tokens2))


def _fuzzy_match(tokens: List[str], target_string: str) -> bool:
    """
    Check if joined tokens contain the target string (fuzzy matching)
    
    Args:
        tokens: List of tokens to join
        target_string: Target string to find
    
    Returns:
        bool: True if target string is found in joined tokens
    """
    # Remove common prefixes (Ġ for GPT-2, ▁ for SentencePiece)
    cleaned_tokens = [
        token.replace('Ġ', ' ').replace('▁', ' ') 
        for token in tokens
    ]
    joined = ''.join(cleaned_tokens)
    
    # Normalize whitespace
    joined_normalized = ' '.join(joined.split())
    target_normalized = ' '.join(target_string.split())
    
    return target_normalized in joined_normalized
