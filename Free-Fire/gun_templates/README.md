# Gun Templates

Place one PNG/JPG per gun here, named by the gun label you want in events.

## File naming

| File | Reported `gun_name` |
|------|---------------------|
| `AK47.png` | `AK47` |
| `M14.png` | `M14` |
| `GROZA.png` | `GROZA` |
| `MP40.png` | `MP40` |
| `AWM.png` | `AWM` |
| `M1887.png` | `M1887` |
| `SCAR.png` | `SCAR` |
| `UMP.png` | `UMP` |

The name is used **exactly as-is** (uppercased) in `gun_name` events, log lines,
and the `siftWeapon` API field.

## How to build your template library

1. Set `"save_unknown_gun_icons": true` in `config.json` (already default).
2. Run the pipeline in a match — unmatched weapon icons are saved to
   `gun_templates/unknown/` automatically (one per 3 seconds, 128×64 px PNG).
3. Open each file in `unknown/`, identify the gun, and rename + move it to
   this folder: e.g.  
   `mv unknown/icon_1718700123_0001.png AK47.png`
4. Restart — or call `reload_gun_templates()` at runtime — to pick up new files.

## Matching tuning

Edit `killfeed/gun_classifier.py` constants if needed:

| Constant | Default | Effect |
|----------|---------|--------|
| `_MATCH_THRESHOLD` | `0.62` | Raise → fewer false positives |
| `_HIST_WEIGHT` | `0.30` | Blend of histogram vs shape score |
| `_ICON_SIZE` | `(64, 32)` | Template resize resolution |
| `_UNKNOWN_SAVE_COOLDOWN` | `3.0 s` | Min gap between unknown saves |

## `unknown/` sub-folder

Auto-generated icon crops when no template matches.  
Use them to grow your template library.  Do **not** put templates here.
