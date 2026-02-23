# Quick start: config + LoRA

## Конфиг EDM QM9

Базовый запуск с пресетом из конфига:

```bash
python main_qm9.py --config configs/edm_qm9.yaml
```

Тот же сетап через пресет:

```bash
python main_qm9.py --preset edm_qm9
```

## LoRA для обоих датасетов

**QM9** (конфиг + LoRA, при необходимости AMP и checkpointing):

```bash
python main_qm9.py --config configs/edm_qm9.yaml --lora_rank 8 --lora_alpha 16
```

С mixed precision и checkpointing:

```bash
python main_qm9.py --config configs/edm_qm9.yaml --lora_rank 8 --use_amp True --use_checkpointing True
```

**GEOM-Drugs** (LoRA, при необходимости AMP и checkpointing):

```bash
python main_geom_drugs.py --n_epochs 3000 --exp_name edm_geom_lora --n_stability_samples 500 \
  --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 \
  --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 4 --lr 1e-4 \
  --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --lora_rank 8 --lora_alpha 16
```

С AMP и checkpointing:

```bash
python main_geom_drugs.py --n_epochs 3000 --exp_name edm_geom_lora --nf 256 --n_layers 4 \
  --batch_size 64 --lr 1e-4 --diffusion_steps 1000 --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --normalize_factors [1,4,10] \
  --ema_decay 0.9999 --lora_rank 8 --use_amp True --use_checkpointing True
```

---

## Скачивание датасетов

- **QM9** — скачивается сам: при первом запуске, если в каталоге `--datadir` (по умолчанию `qm9/temp`) нет готовых сплитов `train.npz` / `valid.npz` / `test.npz`, данные автоматически загружаются с Figshare и обрабатываются.
- **GEOM-Drugs** — автоматического скачивания нет. Нужно один раз подготовить данные по инструкции в `data/geom/README.md` и положить, например, `geom_drugs_30.npy` в `./data/geom/`. Путь к файлу задаётся в `main_geom_drugs.py` (по умолчанию `./data/geom/geom_drugs_30.npy`).
