"""
MultiModal Processor — CLIP-based image/video understanding.

Key fix: Instead of just classifying into generic genre buckets, we now
use a rich vocabulary of movie visual concepts so the generated query is
much more specific and useful for semantic search.
"""

import re
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional
from app.utils.config import settings
from app.utils.logger import logger


# Rich vocabulary of visual concepts relevant to movies
VISUAL_PROMPTS = [
    # Genres / tones
    "a science fiction movie with spaceships",
    "a dystopian science fiction movie",
    "a superhero action movie",
    "a martial arts action movie",
    "a war military movie",
    "a romantic love story movie",
    "a romantic comedy movie",
    "a horror suspense movie",
    "a psychological thriller movie",
    "a crime detective movie",
    "a heist crime movie",
    "an animated children's movie",
    "a fantasy adventure movie",
    "a historical period drama",
    "a biographical drama movie",
    "a family drama movie",
    "a coming of age teen movie",
    "a mystery murder movie",
    "a sports movie",
    "a musical movie",
    "a western movie",
    "a zombie apocalypse movie",
    "a survival disaster movie",
    "a spy espionage movie",
    "a superhero origin story",

    # Visual settings / environments
    "movie scene set in space or on a spaceship",
    "movie scene set in a city at night",
    "movie scene set in a futuristic city",
    "movie scene set in a medieval kingdom",
    "movie scene set in nature or jungle",
    "movie scene set in a post-apocalyptic world",
    "movie scene set in a school",
    "movie scene set in the ocean or underwater",
    "movie scene in a fantasy magical world",
    "movie scene in a haunted house",
    "movie scene at a battlefield or war zone",
    "movie scene in a courtroom",
    "movie scene set in ancient civilization",

    # Visual styles
    "dark atmospheric moody film",
    "bright colorful vibrant film",
    "black and white classic film",
    "animated cartoon film",
    "documentary style film",

    # Emotional tone
    "emotional heartwarming movie",
    "intense action packed movie",
    "lighthearted funny comedy",
    "dark and disturbing movie",
    "inspirational uplifting movie",

    # Character / subject matter
    "movie with robots or artificial intelligence",
    "movie about time travel",
    "movie about magic and wizards",
    "movie about a heist or robbery",
    "movie about a love triangle",
    "movie about family relationships",
    "movie about revenge",
    "movie about a journey or road trip",
    "movie about a musician or artist",
    "movie about a detective or investigation",
    "movie about aliens or extraterrestrials",
    "movie about a virus or pandemic",
    "movie about vampires or werewolves",
    "movie about superheroes",
    "movie about a serial killer",
    "movie about royalty or kingdoms",
    "movie about sports and athletes",
    "movie about war and soldiers",
]


class MultiModalProcessor:
    """
    CLIP-based image/video encoder and concept extractor.
    Converts visual inputs into rich text queries for semantic movie search.
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or settings.CLIP_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")

        try:
            import clip
            self.clip = clip
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            logger.info("CLIP model loaded successfully")

            # Pre-compute all prompt embeddings once
            self._prompt_embeddings = self._precompute_prompt_embeddings()
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.model = None
            self.preprocess = None
            self._prompt_embeddings = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def _precompute_prompt_embeddings(self) -> np.ndarray:
        """Pre-compute text embeddings for all visual prompts."""
        try:
            tokens = self.clip.tokenize(VISUAL_PROMPTS).to(self.device)
            with torch.no_grad():
                text_feats = self.model.encode_text(tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            return text_feats.cpu().float().numpy()
        except Exception as e:
            logger.error(f"Failed to pre-compute prompt embeddings: {e}")
            return None

    def encode_image(self, image: Union[str, Image.Image, np.ndarray]) -> Optional[np.ndarray]:
        """Encode an image to a CLIP feature vector (L2-normalized)."""
        if not self.is_available:
            return None
        try:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                return None

            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_feats = self.model.encode_image(img_tensor)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            return img_feats.cpu().float().numpy()[0]
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def encode_images_batch(self, images: List[Union[str, Image.Image]]) -> Optional[np.ndarray]:
        """Encode multiple images in a single batch."""
        if not self.is_available:
            return None
        try:
            processed = []
            for img in images:
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                processed.append(self.preprocess(img))
            batch = torch.stack(processed).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().float().numpy()
        except Exception as e:
            logger.error(f"Failed to batch encode images: {e}")
            return None

    def image_to_text_query(
        self,
        image: Union[str, Image.Image],
        top_n: int = 5
    ) -> str:
        """
        Generate a rich natural-language query from an image using CLIP.
        Picks top-N matching concepts from VISUAL_PROMPTS and combines them.
        
        Returns a much more specific query than the old approach of just
        picking 3 generic genre labels.
        """
        if not self.is_available or self._prompt_embeddings is None:
            return "movie"

        try:
            img_emb = self.encode_image(image)
            if img_emb is None:
                return "movie"

            # Cosine similarity (already normalized)
            sims = np.dot(self._prompt_embeddings, img_emb)

            # Get top-N matches
            top_idx = np.argsort(sims)[::-1][:top_n]
            top_prompts = [VISUAL_PROMPTS[i] for i in top_idx]
            top_scores = [float(sims[i]) for i in top_idx]

            # Build a rich combined query — strip redundant prefix words
            # and combine the most descriptive parts
            query_parts = []
            seen_words = set()
            for prompt in top_prompts:
                # Remove "movie scene" / "a " / "an " prefixes for cleaner query
                clean = prompt.replace("movie scene ", "").replace("a movie with", "").strip()
                clean = re.sub(r'^(a|an)\s+', '', clean, flags=re.IGNORECASE)
                # Only add if adds new info
                words = set(clean.lower().split())
                if words - seen_words:
                    query_parts.append(clean)
                    seen_words.update(words)

            query = ". ".join(query_parts[:4])
            logger.info(
                f"Generated query (top score: {top_scores[0]:.3f}): {query[:100]}"
            )
            return query if query else "movie"

        except Exception as e:
            logger.error(f"Failed to generate text from image: {e}")
            return "movie"

    def images_to_text_query(
        self,
        images: List[Union[str, Image.Image]],
        top_n: int = 5
    ) -> str:
        """
        Generate a query from multiple frames (for video analysis).
        Averages frame embeddings before matching to prompts.
        """
        if not self.is_available or not images:
            return "movie"

        try:
            embeddings = self.encode_images_batch(images)
            if embeddings is None or len(embeddings) == 0:
                return "movie"

            # Average across frames
            avg_emb = embeddings.mean(axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)

            sims = np.dot(self._prompt_embeddings, avg_emb)
            top_idx = np.argsort(sims)[::-1][:top_n]
            top_prompts = [VISUAL_PROMPTS[i] for i in top_idx]

            query_parts = []
            seen_words = set()
            for prompt in top_prompts:
                clean = prompt.replace("movie scene ", "").replace("a movie with", "").strip()
                clean = re.sub(r'^(a|an)\s+', '', clean, flags=re.IGNORECASE)
                words = set(clean.lower().split())
                if words - seen_words:
                    query_parts.append(clean)
                    seen_words.update(words)

            return ". ".join(query_parts[:4]) or "movie"

        except Exception as e:
            logger.error(f"Failed to process multiple images: {e}")
            return "movie"
