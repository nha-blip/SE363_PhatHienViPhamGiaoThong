"""Step 2 of the 6-class taxonomy: survey what the HCMC 'truck' (3) boxes really are before relabelling them.

Samples 150 HCMC truck boxes from val (full frames) and 150 from train (readable crops), seed 0, short side >= 20 px
(smaller boxes cannot be judged by eye). Each box gets a zero-shot CLIP ViT-L/14 (OpenAI weights, open_clip) suggestion (truck / pickup / minibus /
car). The same CLIP prompts are scored on 300 MIO boxes whose class is known from step 1 (100 each of truck, pickup,
minibus in val) so the suggestion's reliability is measured, not assumed.
Writes <out>/items.json, <out>/crops/<id>.jpg (QA view with a yellow box) and <out>/clip_on_mio.json.
Usage: hcmc_truck_survey_20261125.py <archive root containing vehicle/images> <out dir>
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch
from PIL import Image
import open_clip

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/Vehicle363_6class_labels_20261124"
CLASSES = ["truck", "pickup", "minibus", "car"]
PROMPTS = {
    "truck": ["a photo of a truck", "a photo of a lorry with a cargo box", "a photo of a dump truck"],
    "pickup": ["a photo of a pickup truck", "a photo of a pickup truck with an open cargo bed"],
    "minibus": ["a photo of a passenger van", "a photo of a minibus", "a photo of a cargo van"],
    "car": ["a photo of a car", "a photo of an SUV"],
}


def boxes(label, W, H, cls):
    for i, line in enumerate(label.read_text().splitlines()):
        p = line.split()
        if p and int(p[0]) in cls:
            x, y, w, h = map(float, p[1:5])
            yield i, int(p[0]), [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]


def cut(img, b, pad):
    H, W = img.shape[:2]
    pw, ph = (b[2] - b[0]) * pad, (b[3] - b[1]) * pad
    x1, y1, x2, y2 = int(max(0, b[0] - pw)), int(max(0, b[1] - ph)), int(min(W, b[2] + pw)), int(min(H, b[3] + ph))
    return img[y1:y2, x1:x2], (x1, y1)


def sample(src, split, prefix, cls, n, rng, per_class=False):
    imgs = {p.stem: p for p in (src / "images" / split).iterdir() if p.name.startswith(prefix)}
    cand = []
    for stem in sorted(imgs):
        label = LABELS / split / f"{stem}.txt"
        if not label.exists():
            continue
        with Image.open(imgs[stem]) as im:
            W, H = im.size
        for i, c, b in boxes(label, W, H, cls):
            if min(b[2] - b[0], b[3] - b[1]) >= 20:
                cand.append((imgs[stem], i, c, b))
    rng.shuffle(cand)
    if not per_class:
        return cand[:n]
    out = []
    for c in cls:
        out += [x for x in cand if x[2] == c][:n]
    return out


@torch.no_grad()
def clip_scores(model, proc, text, crops):
    out = []
    for k in range(0, len(crops), 32):
        inp = torch.stack([proc(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops[k:k + 32]]).to("cuda")
        f = model.encode_image(inp).float(); f = f / f.norm(dim=-1, keepdim=True)
        out += (100 * f @ text.T).softmax(-1).tolist()
    return out


def main():
    src, out = Path(sys.argv[1]) / "vehicle", Path(sys.argv[2])
    if out.exists():
        raise FileExistsError(out)
    (out / "crops").mkdir(parents=True)
    rng = random.Random(0)
    hcmc = sample(src, "val", "hcmc__", {3}, 150, rng) + sample(src, "train", "hcmc__", {3}, 150, rng)
    mio = sample(src, "val", "mio__", {3, 4, 5}, 100, rng, per_class=True)
    model = open_clip.load_openai_model(str(Path.home() / ".cache/clip/ViT-L-14.pt"), device="cuda", precision="fp32").eval()
    proc = open_clip.image_transform(model.visual.image_size, is_train=False)
    flat = [(c, p) for c in CLASSES for p in PROMPTS[c]]
    with torch.no_grad():
        tf = model.encode_text(open_clip.get_tokenizer("ViT-L-14")([p for _, p in flat]).to("cuda")).float()
    tf = tf / tf.norm(dim=-1, keepdim=True)
    text = torch.stack([tf[[j for j, (c, _) in enumerate(flat) if c == k]].mean(0) for k in CLASSES])
    text = text / text.norm(dim=-1, keepdim=True)

    def suggest(rows):
        crops = [cut(cv2.imread(str(p)), b, 0.08)[0] for p, _, _, b in rows]
        return [(CLASSES[max(range(4), key=s.__getitem__)], max(s)) for s in clip_scores(model, proc, text, crops)]

    truth = {3: "truck", 4: "pickup", 5: "minibus"}
    conf = Counter((truth[c], g) for (_, _, c, _), (g, _) in zip(mio, suggest(mio)))
    acc = {k: round(sum(v for (t_, g), v in conf.items() if t_ == k and g == k) / 100, 2) for k in truth.values()}
    (out / "clip_on_mio.json").write_text(json.dumps({"accuracy": acc, "confusion": {f"{a}->{b}": v for (a, b), v in sorted(conf.items())}}, indent=1))
    print("CLIP on MIO:", acc, dict(conf))

    items = []
    for k, ((p, i, _, b), (g, s)) in enumerate(zip(hcmc, suggest(hcmc))):
        img = cv2.imread(str(p))
        view, (ox, oy) = cut(img, b, 0.35)
        view = view.copy()
        cv2.rectangle(view, (int(b[0] - ox), int(b[1] - oy)), (int(b[2] - ox), int(b[3] - oy)), (0, 200, 255), 2)
        scale = 360 / max(view.shape[:2])
        if scale < 1:
            view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cid = f"h{k:03d}"
        cv2.imwrite(str(out / "crops" / f"{cid}.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 88])
        items.append({"id": cid, "split": p.parent.name, "image": p.stem, "line": i, "box": [round(v, 1) for v in b],
                      "clip": g, "conf": round(s, 2)})
    (out / "items.json").write_text(json.dumps(items, indent=1))
    print("HCMC suggestions:", Counter(it["clip"] for it in items))


if __name__ == "__main__":
    assert cut(__import__("numpy").zeros((100, 100, 3), "uint8"), [10, 10, 50, 50], 0.5)[1] == (0, 0)
    main()
