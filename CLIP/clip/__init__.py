from .clip import *
from .clip_gradcam import ClipGradcam
import torch
import numpy as np
from PIL import Image
import torchvision
from functools import reduce


def factors(n):
    return set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
        )
    )


saliency_configs = {
    "ours": lambda img_dim: {
        "distractor_labels": {},
        "horizontal_flipping": True,
        "augmentations": 5,
        "imagenet_prompt_ensemble": False,
        "positive_attn_only": True,
        "cropping_augmentations": [
            {"tile_size": img_dim, "stride": img_dim // 4},
            {"tile_size": int(img_dim * 2 / 3), "stride": int(img_dim * 2 / 3) // 4},
            {"tile_size": img_dim // 2, "stride": (img_dim // 2) // 4},
            {"tile_size": img_dim // 4, "stride": (img_dim // 4) // 4},
        ],
    },
    "chefer_et_al": lambda img_dim: {
        "distractor_labels": {},
        "horizontal_flipping": False,
        "augmentations": 0,
        "imagenet_prompt_ensemble": False,
        "positive_attn_only": True,
        "cropping_augmentations": [{"tile_size": img_dim, "stride": img_dim // 4}],
    },
}


class ClipWrapper:
    # SINGLETON WRAPPER
    clip_model = None
    clip_preprocess = None
    clip_gradcam = None
    lavt = None
    device = None
    jittering_transforms = None

    def __init__(self, clip_model_type, device, **kwargs):
        ClipWrapper.device = device
        ClipWrapper.jittering_transforms = torchvision.transforms.ColorJitter(
            brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1
        )
        ClipWrapper.clip_model, ClipWrapper.clip_preprocess = load(
            clip_model_type, ClipWrapper.device, **kwargs
        )
        ClipWrapper.clip_gradcam = ClipGradcam(
            clip_model_name=clip_model_type,
            classes=[""],
            templates=["{}"],
            device=ClipWrapper.device,
            **kwargs
        )

    @classmethod
    def check_initialized(cls, clip_model_type="ViT-B/32", **kwargs):
        if cls.clip_gradcam is None:
            ClipWrapper(
                clip_model_type=clip_model_type,
                device="cuda" if torch.cuda.is_available() else "cpu",
                **kwargs
            )

    @classmethod
    def get_clip_text_feature(cls, string):
        ClipWrapper.check_initialized()
        with torch.no_grad():
            return (
                cls.clip_model.encode_text(
                    tokenize(string, context_length=77).to(cls.device)
                )
                .squeeze()
                .cpu()
                .numpy()
            )

    @classmethod
    def get_visual_feature(cls, rgb, tile_attn_mask, device=None):
        if device is None:
            device = ClipWrapper.device
        ClipWrapper.check_initialized()
        rgb = ClipWrapper.clip_preprocess(Image.fromarray(rgb)).unsqueeze(0)
        with torch.no_grad():
            clip_feature = ClipWrapper.clip_model.encode_image(
                rgb.to(ClipWrapper.device), tile_attn_mask=tile_attn_mask
            ).squeeze()
            return clip_feature.to(device)

    @classmethod
    def get_clip_saliency(
        cls,
        img,
        text_labels,
        prompts,
        distractor_labels=set(),
        use_lavt=False,
        **kwargs
    ):
        cls.check_initialized()
        if use_lavt:
            return cls.lavt.localize(img=img, prompts=text_labels)
        cls.clip_gradcam.templates = prompts
        cls.clip_gradcam.set_classes(text_labels)
        text_label_features = torch.stack(
            list(cls.clip_gradcam.class_to_language_feature.values()), dim=0
        )
        text_label_features = text_label_features.squeeze(dim=-1).cpu()
        text_maps = cls.get_clip_saliency_convolve(
            img=img, text_labels=text_labels, **kwargs
        )
        if len(distractor_labels) > 0:
            distractor_labels = set(distractor_labels) - set(text_labels)
            cls.clip_gradcam.set_classes(list(distractor_labels))
            distractor_maps = cls.get_clip_saliency_convolve(
                img=img, text_labels=list(distractor_labels), **kwargs
            )
            text_maps -= distractor_maps.mean(dim=0)
        text_maps = text_maps.cpu()
        return text_maps, text_label_features.squeeze(dim=-1)

    @classmethod
    def get_clip_saliency_convolve(
        cls,
        text_labels,
        horizontal_flipping=False,
        positive_attn_only: bool = False,
        tile_batch_size=16,
        prompt_batch_size=32,
        tile_interpolate_batch_size=16,
        **kwargs
    ):
        cls.clip_gradcam.positive_attn_only = positive_attn_only
        tiles, tile_imgs, counts, tile_sizes = cls.create_tiles(**kwargs)
        outputs = {
            k: torch.zeros(
                [len(text_labels)] + list(count.shape), device=cls.device
            ).half()
            for k, count in counts.items()
        }
        tile_gradcams = torch.cat(
            [
                torch.cat(
                    [
                        cls.clip_gradcam(
                            x=tile_imgs[tile_idx : tile_idx + tile_batch_size],
                            o=text_labels[prompt_idx : prompt_idx + prompt_batch_size],
                        )
                        for tile_idx in np.arange(0, len(tile_imgs), tile_batch_size)
                    ],
                    dim=1,
                )
                for prompt_idx in np.arange(0, len(text_labels), prompt_batch_size)
            ],
            dim=0,
        )
        if horizontal_flipping:
            flipped_tile_imgs = tile_imgs[
                ..., torch.flip(torch.arange(0, tile_imgs.shape[-1]), dims=[0])
            ]
            flipped_tile_gradcams = torch.cat(
                [
                    torch.cat(
                        [
                            cls.clip_gradcam(
                                x=flipped_tile_imgs[
                                    tile_idx : tile_idx + tile_batch_size
                                ],
                                o=text_labels[
                                    prompt_idx : prompt_idx + prompt_batch_size
                                ],
                            )
                            for tile_idx in np.arange(
                                0, len(tile_imgs), tile_batch_size
                            )
                        ],
                        dim=1,
                    )
                    for prompt_idx in np.arange(0, len(text_labels), prompt_batch_size)
                ],
                dim=0,
            )
            with torch.no_grad():
                flipped_tile_gradcams = flipped_tile_gradcams[
                    ...,
                    torch.flip(
                        torch.arange(0, flipped_tile_gradcams.shape[-1]), dims=[0]
                    ),
                ]
                tile_gradcams = (tile_gradcams + flipped_tile_gradcams) / 2
                del flipped_tile_gradcams
        with torch.no_grad():
            torch.cuda.empty_cache()
            for tile_size in np.unique(tile_sizes):
                tile_size_mask = tile_sizes == tile_size
                curr_size_grads = tile_gradcams[:, tile_size_mask]
                curr_size_tiles = tiles[tile_size_mask]
                for tile_idx in np.arange(
                    0, curr_size_grads.shape[1], tile_interpolate_batch_size
                ):
                    resized_tiles = torch.nn.functional.interpolate(
                        curr_size_grads[
                            :, tile_idx : tile_idx + tile_interpolate_batch_size
                        ],
                        size=tile_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    for tile_idx, tile_slice in enumerate(
                        curr_size_tiles[
                            tile_idx : tile_idx + tile_interpolate_batch_size
                        ]
                    ):
                        outputs[tile_size][tile_slice] += resized_tiles[
                            :, tile_idx, ...
                        ]
            output = sum(
                output.float() / count
                for output, count in zip(outputs.values(), counts.values())
            ) / len(counts)
            del outputs, counts, tile_gradcams
            output = output.cpu()
            return output

    @classmethod
    def create_tiles(cls, img, augmentations, cropping_augmentations, **kwargs):
        assert type(img) == np.ndarray
        images = []
        cls.check_initialized()
        # compute image crops
        img_pil = Image.fromarray(img)
        images.append(np.array(img_pil))
        for _ in range(augmentations):
            images.append(np.array(cls.jittering_transforms(img_pil)))
        # for taking average
        counts = {
            crop_aug["tile_size"]: torch.zeros(img.shape[:2], device=cls.device).float()
            + 1e-5
            for crop_aug in cropping_augmentations
        }
        tiles = []
        tile_imgs = []
        tile_sizes = []
        for img in images:
            for crop_aug in cropping_augmentations:
                tile_size = crop_aug["tile_size"]
                stride = crop_aug["stride"]
                for y in np.arange(0, img.shape[1] - tile_size + 1, stride):
                    if y >= img.shape[0]:
                        continue
                    for x in np.arange(0, img.shape[0] - tile_size + 1, stride):
                        if x >= img.shape[1]:
                            continue
                        tile = (
                            slice(None, None),
                            slice(x, x + tile_size),
                            slice(y, y + tile_size),
                        )
                        tiles.append(tile)
                        counts[tile_size][tile[1:]] += 1
                        tile_sizes.append(tile_size)
                        # this is currently biggest bottle neck
                        tile_imgs.append(
                            cls.clip_gradcam.preprocess(
                                Image.fromarray(img[tiles[-1][1:]])
                            )
                        )
        tile_imgs = torch.stack(tile_imgs).to(cls.device)
        return np.array(tiles), tile_imgs, counts, np.array(tile_sizes)


imagenet_templates = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

__all__ = ["ClipWrapper", "imagenet_templates"]
