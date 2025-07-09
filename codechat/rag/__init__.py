"""RAG chain for code question answering using Gemma model."""

import logging
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_PROMPT_TEMPLATE,
    DEVICE_MAP,
    MAX_CONTEXT_LENGTH,
    MAX_GENERATION_LENGTH,
    RANDOM_SEED,
    TEMPERATURE,
    TOP_K,
    TORCH_DTYPE,
    USE_GPU,
    get_hf_token,
)
from ..embeddings.embedder import CodeEmbedder
from ..ingestion.chunker import CodeChunk
from ..vectordb.faiss_store import FaissStore

logger = logging.getLogger(__name__)


class CodeChat:
    """RAG-based code chat system using Gemma model."""

    def __init__(
        self,
        index: FaissStore,
        model_name: str = DEFAULT_LLM_MODEL,
        prompt_template: str | None = None,
        embedder: CodeEmbedder | None = None,
        use_4bit: bool = True,
    ) -> None:
        """Initialize the CodeChat system.
        
        Args:
            index: FAISS vector store with code embeddings.
            model_name: Name of the language model to use.
            prompt_template: Custom prompt template. Uses default if None.
            embedder: Code embedder instance. Creates new if None.
            use_4bit: Whether to use 4-bit quantization for memory efficiency.
        """
        self.index = index
        self.model_name = model_name
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        
        # Initialize embedder if not provided
        self.embedder = embedder or CodeEmbedder()
        
        # Set random seed
        torch.manual_seed(RANDOM_SEED)
        
        # Configure device and quantization
        self.device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=get_hf_token(),
            trust_remote_code=True,
        )
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure model loading
        model_kwargs = {
            "token": get_hf_token(),
            "trust_remote_code": True,
            "torch_dtype": getattr(torch, TORCH_DTYPE) if self.device == "cuda" else torch.float32,
        }
        
        # Add quantization config if using 4-bit and GPU
        if use_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, TORCH_DTYPE),
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = DEVICE_MAP
        else:
            # Load on specified device without quantization
            model_kwargs["device_map"] = None
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Move to device if not using device_map
        if model_kwargs["device_map"] is None:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Model loaded successfully")

    def chat(self, question: str, top_k: int = TOP_K) -> str:
        """Answer a question about the codebase.
        
        Args:
            question: User question about the code.
            top_k: Number of relevant code chunks to retrieve.
            
        Returns:
            Generated answer based on retrieved code context.
        """
        # Retrieve relevant code chunks
        relevant_chunks = self._retrieve_relevant_chunks(question, top_k)
        
        if not relevant_chunks:
            return "I don't know based on the provided code context."
        
        # Format context
        context = self._format_context(relevant_chunks)
        
        # Generate answer
        answer = self._generate_answer(question, context)
        
        return answer

    def _retrieve_relevant_chunks(self, question: str, top_k: int) -> List[Tuple[CodeChunk, float]]:
        """Retrieve relevant code chunks for a question.
        
        Args:
            question: User question.
            top_k: Number of chunks to retrieve.
            
        Returns:
            List of (CodeChunk, similarity_score) tuples.
        """
        # Embed the question
        question_embedding = self.embedder.embed_single(question)
        
        # Query the index
        results = self.index.query(question_embedding, k=top_k)
        
        # Filter results with reasonable similarity threshold
        filtered_results = [(chunk, score) for chunk, score in results if score > 0.1]
        
        return filtered_results

    def _format_context(self, relevant_chunks: List[Tuple[CodeChunk, float]]) -> str:
        """Format retrieved chunks into context string.
        
        Args:
            relevant_chunks: List of (CodeChunk, similarity_score) tuples.
            
        Returns:
            Formatted context string.
        """
        context_parts = []
        
        for i, (chunk, score) in enumerate(relevant_chunks, 1):
            context_part = f"### Code Snippet {i} (Similarity: {score:.3f})\n"
            context_part += f"File: {chunk.file_path}\n"
            context_part += f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            context_part += f"Language: {chunk.language}\n"
            context_part += f"Type: {chunk.chunk_type}\n\n"
            context_part += f"```{chunk.language}\n{chunk.content}\n```\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using the language model.
        
        Args:
            question: User question.
            context: Retrieved code context.
            
        Returns:
            Generated answer.
        """
        # Format prompt
        prompt = self.prompt_template.format(context=context, question=question)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CONTEXT_LENGTH,
            padding=True,
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_GENERATION_LENGTH,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove the prompt part)
        answer = response[len(prompt):].strip()
        
        return answer

    def get_model_info(self) -> dict:
        """Get information about the loaded models.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "llm_model": self.model_name,
            "device": str(self.device),
            "embedder": self.embedder.get_model_info(),
            "index_stats": self.index.get_stats(),
        }
