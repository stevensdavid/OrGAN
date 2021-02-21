from PIL import Image


def pad_to_square(pil_image: Image):
    """Adapted from https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/"""
    w, h = pil_image.size
    side = max(w, h)
    if w == h:
        return pil_image
    # pad with black
    result = Image.new(pil_image.mode, (side, side), (0, 0, 0))
    if w > h:
        result.paste(pil_image, (0, (w - h) // 2))
    else:
        result.paste(pil_image, ((h - w) // 2, 0))
    return result
