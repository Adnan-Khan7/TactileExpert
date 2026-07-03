"""GPT-image-1 wrapper for tactile graphic generation and editing.

API key is accepted as a parameter on every call — never stored, never read
from environment variables. The caller is responsible for prompting the user.
"""

import base64
import io
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# BANA-grounded default prompt template
# ---------------------------------------------------------------------------

BANA_TEMPLATE = """\
Create a tactile graphic of {object_description} suitable for embossing on swell paper \
for blind and visually impaired users.

The tactile graphic MUST be faithful to the reference photograph: preserve the exact \
viewpoint, overall proportions, and structural layout of the subject as it appears \
in the image. Do not substitute a generic or canonical depiction of the object.

BANA guidelines to follow:
- Draw ONLY the subject object — remove all background scenery, environment, and context. \
The background must be plain white with no texture, shadow, or surrounding elements.
- The complete object must be fully visible within the image boundaries — do not crop, \
clip, or cut off any part of the subject. Leave adequate white margin on all four sides.
- Use bold, continuous black outlines with no gaps or breaks (minimum 1.5 mm stroke width \
when embossed at standard resolution)
- Keep all adjacent lines separated by at least 1.5 mm so a fingertip can distinguish them
- Apply distinct tactile texture fills (fine dots, diagonal lines, crosshatch, or stipple) \
to differentiate surfaces or regions that appear visually distinct in the reference photograph
- Include all significant anatomical or structural parts visible in the reference image — \
do not simplify to the point of omission
- No shadows, no gradients, no colour fills — black outlines and texture patterns on white only
- Avoid decorative elements, labels, or fine detail that would merge or become tactilely \
ambiguous at embossed resolution

Render the object from the same viewpoint as the reference photograph. \
Capture the essential silhouette and the key internal structural features \
that a person would identify through touch.\
"""


def build_prompt(object_description: str) -> str:
    """Fill the BANA template with the object description."""
    return BANA_TEMPLATE.format(object_description=object_description.strip())


def _png_buffer(img: Image.Image, name: str) -> tuple[str, io.BytesIO, str]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return (name, buf, "image/png")


def _pad_to_square(img: Image.Image) -> Image.Image:
    """Pad a non-square image to square with white background.

    Prevents the API from cropping or distorting the subject when the reference
    photo has a non-1:1 aspect ratio.
    """
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


# ---------------------------------------------------------------------------
# Generation (call 1 — create from scratch)
# ---------------------------------------------------------------------------

def generate_tactile(
    api_key: str,
    prompt: str,
    natural_reference: Image.Image | None = None,
    size: str = "1024x1024",
    quality: str = "medium",
) -> Image.Image:
    """
    Call GPT-image-1 to generate a tactile graphic.

    When `natural_reference` is provided the image is sent via images.edit so
    GPT grounds its output in the actual photograph rather than imagining a
    generic depiction from the text description alone.  This is the preferred
    path and fixes faithfulness issues with low-resolution or ambiguous inputs.

    Parameters
    ----------
    api_key           : OpenAI API key — entered by user, used once, not stored.
    prompt            : Full generation prompt (typically BANA_TEMPLATE filled in).
    natural_reference : Natural reference photograph as PIL image (strongly recommended).
    size              : Image size string accepted by the API.
    quality           : 'low', 'medium', or 'high'.

    Returns
    -------
    PIL.Image   : Generated image.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    if natural_reference is not None:
        ref_file = _png_buffer(_pad_to_square(natural_reference.convert("RGB")), "reference.png")
        response = client.images.edit(
            model="gpt-image-1",
            image=ref_file,
            prompt=(
                "The image provided is a natural reference photograph. "
                "Convert it into a tactile graphic — do NOT edit the photograph itself. "
                "Stay strictly faithful to this specific image: preserve the exact viewpoint, "
                "proportions, pose, and structural layout of the subject as shown. "
                "Do not substitute a generic or canonical depiction of the object.\n\n"
                + prompt
            ),
            size=size,
            n=1,
        )
    else:
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )

    img_bytes = base64.b64decode(response.data[0].b64_json)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ---------------------------------------------------------------------------
# Editing (call 2 — refine existing candidate)
# ---------------------------------------------------------------------------

def edit_tactile(
    api_key: str,
    current_tactile: Image.Image,
    edit_instruction: str,
    natural_reference: Image.Image | None = None,
    size: str = "1024x1024",
) -> Image.Image:
    """
    Call GPT-image-1 in edit mode to refine an existing tactile graphic.

    Parameters
    ----------
    api_key           : OpenAI API key — entered by user, used once, not stored.
    current_tactile   : The current tactile graphic as a PIL image.
    edit_instruction  : Plain-language edit instruction (from evaluator or user).
    natural_reference : Optional natural reference photograph. When provided it
                        is sent alongside the tactile graphic so the edit is
                        grounded in the real object (user-toggled in the UI).
    size              : Image size string accepted by the API.

    Returns
    -------
    PIL.Image   : Edited image.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    images = [_png_buffer(current_tactile, "tactile.png")]
    if natural_reference is not None:
        images.append(_png_buffer(_pad_to_square(natural_reference.convert("RGB")), "reference.png"))
        prompt = (
            "The first image is a tactile graphic intended for embossing on swell "
            "paper; the second image is the natural reference photograph of the "
            "same object. Edit the FIRST image only. Use the reference photograph "
            "to verify shapes, parts, and surface regions. Apply the following "
            f"correction while preserving all other features: {edit_instruction}"
        )
    else:
        prompt = (
            "This is a tactile graphic intended for embossing on swell paper. "
            f"Apply the following correction while preserving all other features: {edit_instruction}"
        )

    response = client.images.edit(
        model="gpt-image-1",
        image=images if natural_reference is not None else images[0],
        prompt=prompt,
        size=size,
        n=1,
    )
    img_bytes = base64.b64decode(response.data[0].b64_json)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
