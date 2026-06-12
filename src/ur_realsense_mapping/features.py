# SPDX-License-Identifier: Apache-2.0
"""C-RADIO feature extraction and text-aligned queries.

Adapted from cuRobo's ``examples/getting_started/feature_mapping.py``
(NVIDIA, Apache-2.0). Downloads the RADIO checkpoint through ``torch.hub``
(NVlabs/RADIO) on first use; export ``HF_TOKEN`` if Hugging Face auth is
needed. Extra deps: ``timm transformers torchvision einops``.
"""

import torch
import torch.nn.functional as F

RADIO_MODEL_NAME = "c-radio_v3-B"
TEXT_ADAPTOR_NAME = "siglip2"


class RadioFeatures:
    """C-RADIO patch features for mapping + SigLIP text encoding for queries."""

    def __init__(
        self,
        model_name: str = RADIO_MODEL_NAME,
        device: str = "cuda:0",
        text_adaptor_name: "str | None" = TEXT_ADAPTOR_NAME,
    ):
        self.device = device
        self.text_adaptor_name = text_adaptor_name
        adaptor_names = [text_adaptor_name] if text_adaptor_name else None

        self.model = (
            torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                source="github",
                version=model_name,
                progress=True,
                skip_validation=True,
                adaptor_names=adaptor_names,
            )
            .eval()
            .to(device)
        )
        self.patch_size = int(getattr(self.model, "patch_size", 16))

        self.text_adaptor = None
        self.tokenizer = None
        self._encode_text = None
        if text_adaptor_name is not None:
            self.text_adaptor = self._resolve_text_adaptor(text_adaptor_name)
            self.tokenizer = getattr(self.text_adaptor, "tokenizer", None)
            self._encode_text = getattr(self.text_adaptor, "encode_text", None)
            if self.tokenizer is None or not callable(self._encode_text):
                raise RuntimeError(
                    f"RADIO adaptor '{text_adaptor_name}' must expose tokenizer "
                    "and encode_text for text matching."
                )

    @torch.inference_mode()
    def extract_patch_features(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        """``(H, W, 3)`` uint8 on ``self.device`` -> ``(H_p, W_p, D)`` float32."""
        H, W = rgb_uint8.shape[:2]
        target_h, target_w = self.model.get_nearest_supported_resolution(H, W)

        img = rgb_uint8.permute(2, 0, 1).float() / 255.0
        img = F.interpolate(
            img.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        output = self.model(img)
        if isinstance(output, dict):
            output = output["backbone"]
        features = getattr(output, "features", None)
        if features is None:
            _, features = output  # (summary, features)

        ps = self.patch_size
        return features[0].view(target_h // ps, target_w // ps, -1).contiguous()

    def _resolve_text_adaptor(self, adaptor_name: str):
        for attr in ("adaptors", "adapters", "_adaptors"):
            registry = getattr(self.model, attr, None)
            if registry is not None and adaptor_name in registry:
                return registry[adaptor_name]
        raise RuntimeError(f"Could not find adaptor '{adaptor_name}' on RADIO model.")

    def _project_through_text_adaptor(self, features: torch.Tensor) -> torch.Tensor:
        # Adaptor APIs vary across checkpoint variants; try common entry points.
        for attr in ("head_mlp", "feat_mlp", "head"):
            sub = getattr(self.text_adaptor, attr, None)
            if sub is not None and callable(sub):
                return sub(features)
        if callable(self.text_adaptor):
            try:
                out = self.text_adaptor(features)
            except TypeError:
                summary = features.mean(dim=0, keepdim=True)
                out = self.text_adaptor(summary, features.unsqueeze(0))
                if isinstance(out, tuple):
                    out = out[1]
                    if out.dim() == 3:
                        out = out[0]
            if isinstance(out, tuple):
                out = out[1] if len(out) > 1 else out[0]
            return out
        raise RuntimeError("RADIO text adaptor has no known projection entry point.")

    @torch.inference_mode()
    def encode_text(self, text) -> torch.Tensor:
        """One or more strings -> ``(N, D_teacher)`` L2-normalized features."""
        if self.tokenizer is None or self._encode_text is None:
            raise RuntimeError("Loaded RADIO model has no text tower.")
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text)
        if hasattr(tokens, "to"):
            tokens = tokens.to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                feats = self._encode_text(tokens, normalize=True)
            except TypeError:
                feats = F.normalize(self._encode_text(tokens), dim=-1)
        return feats

    @torch.inference_mode()
    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """``(N, D_radio)`` map features -> L2-normalized teacher-space features."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.normalize(self._project_through_text_adaptor(features), dim=-1)
