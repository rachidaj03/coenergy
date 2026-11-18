# coEnergy — assessment (HLC, Hb, CDT à partir de T_int(t) et U(t))

## Structure
- `src/coenergy/` : modules (`data`, `eda`, `features`, `models`, `train`, `evaluate`, `config`, `loader`)
- `notebooks/assessment_coenergy.ipynb` : **notebook de rendu** (importe les modules)
- `notebooks/assessment_coenergy_colab.ipynb` : variante autonome (code inclus) pour Colab, générée par `notebooks/build_colab.py`
- `tests/` : tests unitaires (contraintes physiques, décodage des cibles)
- `pyproject.toml`, `uv.lock` : environnement reproductible (`uv`)

## Exécution locale
```bash
uv sync
uv run pytest
uv run jupyter lab notebooks/assessment_coenergy.ipynb   # depuis notebooks/
```

## Colab (GPU) — notebook de rendu
Téléverser dans `/content` : `dist/coenergy_src.zip` (le notebook le décompresse) ; les données sont lues depuis Google Drive
(`DATA_DIR`). Périphérique : CUDA détecté automatiquement ; pour un MacBook, décommenter les lignes MPS de
`src/coenergy/config.py`.

## Colab — notebook autonome
Téléverser uniquement `util_assessement.py` ; ouvrir `notebooks/assessment_coenergy_colab.ipynb`.
