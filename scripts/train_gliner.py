"""GLiNER'i başka bir makinede (ör. GPU'lu Colab) eğitmek için bağımsız betik.

Kullanım:
  pip install gliner==0.2.29 accelerate
  python train_gliner.py <proje>-gliner-train.json --base urchade/gliner_multi-v2.1 --epochs 5 --out model
  (cd model && zip -r ../model.zip .)   # sonra web arayüzünde "modeli yükle"

Girdi: web arayüzündeki "GLiNER eğitim biçiminde indir" dosyası ({"labels", "train", "test"}).
"""
import argparse
import json
import math

import torch
from gliner import GLiNER


def print_eval(model, test, labels, title):
    """Farklı eşiklerde tam eşleşme F1 (en iyisini web arayüzünde 'eşik' olarak kullan)."""
    print(f"--- {title} ---")
    best = (0.0, 0.5)
    for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        _, f1 = model.evaluate(test, flat_ner=True, threshold=th, entity_types=labels)
        best = max(best, (float(f1), th))
        print(f"eşik {th}: F1 {float(f1):.3f}")
    print(f"en iyi: eşik {best[1]} → F1 {best[0]:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--base", default="urchade/gliner_multi-v2.1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--others-lr", type=float, default=5e-5)
    ap.add_argument("--out", default="model")
    a = ap.parse_args()

    data = json.load(open(a.data, encoding="utf-8"))
    train, test, labels = data["train"], data.get("test") or [], data["labels"]
    model = GLiNER.from_pretrained(a.base)
    if test:
        print_eval(model, test, labels, "temel model (eğitimsiz)")
    steps = max(1, a.epochs * math.ceil(len(train) / a.batch))
    model.train_model(
        train_dataset=train, eval_dataset=test or None, output_dir="_checkpoints",
        learning_rate=a.lr, others_lr=a.others_lr, weight_decay=0.01, others_weight_decay=0.01,
        focal_loss_alpha=0.75, focal_loss_gamma=2, warmup_ratio=0.1, per_device_train_batch_size=a.batch,
        max_steps=steps, save_strategy="no", eval_strategy="no", logging_steps=max(1, steps // 20),
        use_cpu=not torch.cuda.is_available(), dataloader_num_workers=0, report_to="none")
    model.save_pretrained(a.out)
    if test:
        print_eval(model, test, labels, "eğitilmiş model")
    print(f"model kaydedildi: {a.out}")


if __name__ == "__main__":
    main()
