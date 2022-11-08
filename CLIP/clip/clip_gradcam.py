from typing import List
import torch
import torch.nn as nn
from .clip_explainability import load
from .clip import tokenize
from torch import device
import numpy as np
import torch.nn.functional as nnf
import itertools


def zeroshot_classifier(clip_model, classnames, templates, device):
    with torch.no_grad():
        texts = list(
            itertools.chain(
                *[
                    [template.format(classname) for template in templates]
                    for classname in classnames
                ]
            )
        )  # format with class
        texts = tokenize(texts).to(device)  # tokenize
        class_embeddings = clip_model.encode_text(texts)
        class_embeddings = class_embeddings.view(len(classnames), len(templates), -1)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        zeroshot_weights = class_embeddings.mean(dim=1)
        return zeroshot_weights.T  # shape: [dim, n classes]


class ClipGradcam(nn.Module):
    def __init__(
        self,
        clip_model_name: str,
        classes: List[str],
        templates: List[str],
        device: device,
        num_layers=10,
        positive_attn_only=False,
        **kwargs
    ):

        super(ClipGradcam, self).__init__()
        self.clip_model_name = clip_model_name
        self.model, self.preprocess = load(clip_model_name, device=device, **kwargs)
        self.templates = templates
        self.device = device
        self.target_classes = None
        self.set_classes(classes)
        self.num_layers = num_layers
        self.positive_attn_only = positive_attn_only
        self.num_res_attn_blocks = {
            "ViT-B/32": 12,
            "ViT-B/16": 12,
            "ViT-L/14": 16,
            "ViT-L/14@336px": 16,
        }[clip_model_name]

    def forward(self, x: torch.Tensor, o: List[str]):
        """
        non-standard hack around an nn, really should be more principled here
        """
        image_features = self.model.encode_image(x.to(self.device))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        zeroshot_weights = torch.cat(
            [self.class_to_language_feature[prompt] for prompt in o], dim=1
        )
        logits_per_image = 100.0 * image_features @ zeroshot_weights
        return self.interpret(logits_per_image, self.model, self.device)

    def interpret(self, logits_per_image, model, device):
        # modified from: https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb#scrollTo=fWKGyu2YAeSV
        batch_size = logits_per_image.shape[0]
        num_prompts = logits_per_image.shape[1]
        one_hot = [logit for logit in logits_per_image.sum(dim=0)]
        model.zero_grad()

        image_attn_blocks = list(
            dict(model.visual.transformer.resblocks.named_children()).values()
        )
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(
            num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype
        ).to(device)
        R = R[None, None, :, :].repeat(num_prompts, batch_size, 1, 1)
        for i, block in enumerate(image_attn_blocks):
            if i <= self.num_layers:
                continue
            # TODO try scaling block.attn_probs by value magnitude
            # TODO actual parallelized prompt gradients
            grad = torch.stack(
                [
                    torch.autograd.grad(logit, [block.attn_probs], retain_graph=True)[
                        0
                    ].detach()
                    for logit in one_hot
                ]
            )
            grad = grad.view(
                num_prompts,
                batch_size,
                self.num_res_attn_blocks,
                num_tokens,
                num_tokens,
            )
            cam = (
                block.attn_probs.view(
                    1, batch_size, self.num_res_attn_blocks, num_tokens, num_tokens
                )
                .detach()
                .repeat(num_prompts, 1, 1, 1, 1)
            )
            cam = cam.reshape(num_prompts, batch_size, -1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(
                num_prompts, batch_size, -1, grad.shape[-1], grad.shape[-1]
            )
            cam = grad * cam
            cam = cam.reshape(
                num_prompts * batch_size, -1, cam.shape[-1], cam.shape[-1]
            )
            if self.positive_attn_only:
                cam = cam.clamp(min=0)
            # average of all heads
            cam = cam.mean(dim=-3)
            R = R + torch.bmm(
                cam, R.view(num_prompts * batch_size, num_tokens, num_tokens)
            ).view(num_prompts, batch_size, num_tokens, num_tokens)
        image_relevance = R[:, :, 0, 1:]
        img_dim = int(np.sqrt(num_tokens - 1))
        image_relevance = image_relevance.reshape(
            num_prompts, batch_size, img_dim, img_dim
        )
        return image_relevance

    def set_classes(self, classes):
        self.target_classes = classes
        language_features = zeroshot_classifier(
            self.model, self.target_classes, self.templates, self.device
        )

        self.class_to_language_feature = {}
        for i, c in enumerate(self.target_classes):
            self.class_to_language_feature[c] = language_features[:, [i]]
