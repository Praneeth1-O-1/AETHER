# AETHER — Phase 5 Data Acquisition Specification

**For:** the `aether-dataset` acquisition pipeline (a separate repo/session).
**Purpose:** hand-off spec for the three re-acquisition tasks in
`AETHER_SAR_Cloud_Remediation_Report.docx` §6, Phase 5 — the highest-ceiling,
budget-gated phase, which needs genuinely new downloads rather than a code fix
in the `AETHER` training repo.

**Do not start here.** Phases 0–3 (fix the fusion measurement bug, inject
synthetic cloud, fix SAR confounds, force per-modality prediction) are
code-only, run first, and may make some of this unnecessary. This document
exists so the acquisition side can be scoped and costed in parallel, not so it
ships before Phase 0–3 land.

---

## 1. Baseline — what already exists (do not re-derive, do not re-download)

Measured 2026-09-06/07 by scanning all 6,406 `meta.json` files in `data/strict`
plus direct raster statistics over an 80–500 tile sample.

| Property | Value |
|---|---|
| Tiles | 6,406 (256×256 px, 10 m/px, 15 input bands, 5 label bands) |
| AOIs / distinct geographic locations | 544 / **544** (zero repeat-visit locations — every AOI is a single year, single place) |
| Countries | 22 |
| Total size | 16 GB |
| AOI-year distribution | 2017:1, 2018:4, 2019:4, 2020:10, 2021:58, 2022:26, **2023:441** |
| Non-2023 AOIs | 103 / 544 (19%) — but these are 103 *different places*, not revisits |
| Optical `valid_fraction` ≥ 0.95 | 95.5% of tiles (89.4% exactly 1.0) |
| Optical meaningfully cloud-occluded (`valid_fraction` < 0.95) | **277 tiles (4.5%)** |
| Optical blank (`valid_fraction` = 0, S2 record present) | 11 tiles (0.18%) |
| Sentinel-1 dates per tile | **1** (single acquisition, no temporal stack) |
| Sentinel-1 orbit split | 73.8% DESCENDING (4,727) / 26.2% ASCENDING (1,678), 89 distinct relative orbits |
| Countries single-orbit-direction | 17 of 22 (orbit is a geography proxy) |
| \|S1 offset from S2 anchor\| | median 3 d, mean 3.31 d, max 14 d; 73.4% within 4 d, 5.1% at 10–14 d |
| SAR-only LULC accuracy (measured, this archive) | 56.4% |
| SAR-only LULC accuracy (literature, multi-temporal stacks) | 74–84% |

**Implication that shapes every request below:** this archive cannot supply
seasonal/multi-temporal SAR by re-slicing what's already downloaded — the
data genuinely isn't there. Every location was visited once.

---

## 2. What the acquisition tool can already do (checked against `aether-dataset` source, not assumed)

Relevant knobs that exist today in `config/pipeline.*.yaml` and
`scripts/plan_snapshots.py`:

- `sentinel1.orbit`: `ASCENDING | DESCENDING` — **one, pinned per AOI, for the
  whole series.** The tool's own comment: *"NOT BOTH. Ascending and descending
  view the same ground from opposite sides; backscatter off a slope or wall
  differs by geometry, not noise, and no reducer reconciles it."* This is a
  physical decision, not an oversight — **do not ask for both orbits per AOI**;
  ask for an incidence-angle raster instead (Task 3).
- `sentinel1.selection`: `nearest | median_composite` — **one scene per tile
  today.** There is no "N seasonal dates per AOI" mode. Task 1 below is a new
  capability, not a config flag.
- `sentinel1.max_offset_days`: 14, rejected (not substituted) beyond that.
- `sentinel2.max_cloud_cover` / `cloud_score_plus_threshold`: scene-level
  filters used to find the *clearest* anchor. There is no "give me the
  cloudiest usable scene in this window" mode — Task 3 needs that added.
- `scripts/plan_snapshots.py --snapshots-per-aoi`: produces **one AOI-year
  entry per year**, i.e. annual revisits (what built the 103 non-2023 AOIs
  as separate places, historically also used for true same-place revisits in
  `outputs_all_series`). This is year-level, not intra-season — not what
  Task 1 needs.
- Every AOI outside the Open Buildings Temporal footprint (Boreal Forest,
  Tundra, most of Temperate Conifer) can only ever carry 3 of 5 label bands.
  Any new AOI selection should stay inside that footprint unless the goal is
  specifically LULC/road-only tiles.

---

## 3. Task 1 — Multi-temporal Sentinel-1 (highest priority, highest ceiling)

**What:** 3–4 additional Sentinel-1 acquisitions per selected location,
spread across the growing season, to give the SAR encoder backscatter
phenology instead of one frozen date. This is what the literature's
74–84% SAR-only figures depend on and what this archive's 56.4% is missing.

**New capability needed:** a query mode that, for a given AOI + existing
anchor date, returns the nearest qualifying S1 scene inside each of several
sub-windows across ±3 months of the anchor (e.g. −90d, −30d, +30d, +90d),
each independently subject to the existing `max_offset_days`-style coverage
check (≥95% AOI coverage), same orbit direction as the AOI's existing pin.
`meta.json` already records that multiple qualifying S1 scenes existed per
window (see any tile's `acquisition.sentinel1.note`), so the scenes likely
exist in the catalog already — this is a query-shape change, not a new
sensor requirement.

**Scope (proposed, budget-gated — not "all 544 locations"):**

| | |
|---|---|
| Locations | 100–150, stratified to preserve current country/biome mix and orbit-direction balance (do not just take the first N) |
| Dates per location | 4 (matches Table 16's "3–4 seasonal dates") |
| New S1 scenes | 400–600 |
| Reuse | Optical, DEM, and all label bands are unchanged and must NOT be re-downloaded — only new Sentinel-1 is needed per existing tile |
| Estimated new storage | ~15 GB (Table 16, §5.1) |

**Acceptance / quality gates to report per acquired scene** (mirror the
existing `meta.json` schema so tiles stay drop-in compatible):

- `offset_days` from each sub-window's target date, and from the original anchor
- AOI coverage fraction (reject < 95%, same as current `max_offset_days` rule)
- Orbit direction and relative orbit number (must match the AOI's existing pin)
- Count of qualifying candidate scenes considered per sub-window (for audit)

**Do not do:** request both orbit directions, request a different tile grid
than the existing one, or touch any label band.

---

## 4. Task 2 — Local incidence-angle raster (cheap, do alongside Task 1)

**What:** one new per-tile raster: local incidence angle in degrees, derived
from the *already-recorded* Sentinel-1 granule/orbit geometry for every
existing tile (all 6,406, not just the Task-1 subset).

**Why:** converts the current ASC/DESC confound (Table 6: 26/74 split, 17 of
22 countries single-direction) from an unexplainable, geography-correlated
variance source into a physically modelled one. This is the correct fix for
D3 — pinning one orbit per AOI is deliberate and correct on the acquisition
side (§2 above); the model just needs to be told the angle instead.

**Scope:**

| | |
|---|---|
| Tiles | all 6,406 existing tiles |
| New acquisitions | **none** — computed from each tile's already-recorded `acquisition.sentinel1.scene_id` / orbit metadata, no new Earth Engine imagery pull |
| Estimated new storage | small (one extra float32 band × 6,406 tiles) |

If local incidence angle isn't directly obtainable from the orbit file for a
given scene, a coarse ASC/DESC ± nominal angle proxy is an acceptable
fallback — record which method was used per tile.

---

## 5. Task 3 — Real cloudy Sentinel-2, acquired fresh, targeted by cloud climatology

**Supersedes the "reuse already-queried windows" framing below.** Reusing the
existing 544 AOIs' already-picked windows caps this task at roughly the ~277
tiles already flagged `valid_fraction < 0.95` — that number is an artifact of
this archive's own clear-anchor selection, not a limit on how much real cloud
exists. Querying Earth Engine fresh, aimed at the right places and seasons,
removes that cap.

**The one constraint that fresh acquisition does NOT remove:** a cloudy
Sentinel-2 granule does not supply its own label. Dynamic World is generated
per-granule, so a cloudy scene's own DW output is unreliable exactly under
the cloud — the one place ground truth is needed most. **Every real-cloud
tile still needs a temporally-close CLEAR scene of the same ground** to
inherit a valid label; that pairing is physics, not a policy choice. What
changes is that both the clear anchor and the cloudy scene can now be newly
acquired at new locations and seasons — not limited to the current archive's
existing anchors or windows.

**Where real cloud is actually abundant — measured, not assumed.** Scanning
this archive's own already-*clear*-selected anchors, scene-level cloud cover
(the parent granule's `CLOUDY_PIXEL_PERCENTAGE`, before any clear-scene
filtering) still varies by two orders of magnitude by region:

| Region (existing AOIs) | Mean scene cloud % (of the picked-clear anchor) |
|---|---|
| Ghana | 18.3% |
| Papua (New Guinea) | 16.1% |
| Vietnam | 10.8% |
| north-eastern zone | 8.6% |
| Egypt | 7.5% |
| Cuba | 7.0% |
| Ethiopia | 0.75% |
| Norte (Andean) | 0.71% |
| Argentina | 2.0% |
| Bolivia | 2.4% |

Even the *best available* scene in Ghana/Papua/Vietnam carries 4–7x the cloud
of the best available scene in Ethiopia/Argentina/Bolivia. That is regional
climate, not noise — a fresh wet-season pull in the first group will find
real cloud easily; the same effort in the second group will not, because
those places genuinely don't have much cloud to find. This is not a gap to
fill — an AOI that is almost never cloudy does not need cloud-robust
training, because cloud is not a real-world risk there.

**New acquisition design:**

1. Prioritize existing AOIs in Ghana, Papua, Vietnam, and secondarily Cuba,
   Egypt, north-eastern-zone — the regions already shown above to carry real
   cloud climatology — over the archive's other 16 regions.
2. For each prioritized location, query the **full available Sentinel-2
   history for that AOI** (not the existing 14–30 day anchor window), aimed
   at that region's wet season, and pick a clear scene to serve as a fresh
   label anchor.
3. Around that anchor, pull every genuinely cloudy Sentinel-2 pass within
   the same wet-season window (5-day revisit; a 2–3 month window yields
   ~12–18 candidate dates) rather than a single alternate scene.
4. Where the existing 544 AOIs in these regions are too few, add new AOIs in
   the same regions chosen specifically for wet-season cloud frequency —
   this is a deliberate expansion, not a reuse of what's already queried.
5. Do NOT spend acquisition budget chasing real cloud in Algeria, Egypt's
   interior, Tunisia, or the arid halves of Argentina/Bolivia/Chile/Pakistan
   — the measurement above shows it isn't there to find.

**Scope (revised):**

| | |
|---|---|
| Target regions | Ghana, Papua, Vietnam primary; Cuba, Egypt, north-eastern-zone secondary |
| Locations | existing AOIs in those regions, plus new ones in the same regions if too few exist |
| Cloudy scenes per location | as many as the wet-season window actually contains (climate-limited, not query-limited) — expect several per location, not one |
| Target composition | a spread across occlusion levels — roughly 10–20%, 30–50%, 60–80%, and >80% — so the degradation curve (report §2, "headline deliverable") has real anchor points at each level |
| Label reuse | LULC label always derives from that location's own clear anchor (existing or freshly picked) — a cloudy scene is an **additional optical input variant for the same label**, never a label source itself |
| Estimated volume | plausibly low thousands of tiles if scoped to the primary regions across a full wet season, versus the ~277-tile ceiling of reusing existing windows |
| Estimated new storage | ~8 GB was Table 16's estimate for the narrow reuse version; revise upward in proportion to however many locations/dates this expanded scope actually covers |

**Acceptance / quality gates to report per pair:**

- Cloud cover / valid_fraction of the new cloudy scene (scene-level AND tile-level — the report's §3.1 finding is that these two numbers diverge substantially, so both must be recorded, not just scene-level `CLOUDY_PIXEL_PERCENTAGE`)
- Days offset between the cloudy scene and its clear anchor (flag anything beyond ~60 days — real land-cover change risk grows with the gap)
- Confirmation the LULC label points at the clear anchor's own granule (i.e. `label_sources.lulc` matches that anchor, not the cloudy scene)
- Region and season recorded per tile, so downstream training/eval can stratify by cloud climatology rather than treating "real cloud" as one undifferentiated bucket

---

## 6. Output format — must stay drop-in compatible

Whatever comes back must fit the reader in `AETHER/data/dataset.py` without a
schema migration:

- Same tile grid: 256×256 px, 10 m/px, UTM auto per-AOI (`utm_auto`), origin-snapped
- New bands appended to `inputs.tif` (never inserted mid-list — band order is
  positional in the training repo), each with its own `*_valid` companion band
  where partial validity is possible
- Every new/changed field recorded in `meta.json` under `acquisition` and
  `alignment`, matching the existing per-band note style (source scene id,
  offset_days, coverage fraction, a human-readable `note` string) — this
  archive's entire quality-audit trail depends on that convention
- Nothing overwrites an existing band or existing tile's existing files;
  additions are new bands / new sibling files, so the current `checkpoints/*`
  and `outputs/*` results remain reproducible against the untouched originals

## 7. Priority order

1. **Task 1** (multi-temporal SAR) — gates the single biggest number in the
   report (SAR-only accuracy 56.4% → target 74–84%). Everything else is
   secondary to this.
2. **Task 2** (incidence angle) — cheap, no new downloads, do it regardless of
   whether Task 1 or 3 proceed.
3. **Task 3** (real cloudy scenes, freshly acquired) — no longer just a
   validation slice for synthetic cloud injection. At the revised scope
   (§5), targeted at Ghana/Papua/Vietnam and secondarily Cuba/Egypt/
   north-eastern-zone, this can plausibly reach low-thousands of genuine
   (cloudy input, valid label) tiles — large enough to be a real training
   source for those regions' cloud robustness, not only a sanity check.
   Synthetic injection (Phase 1, code repo) still fills two gaps real cloud
   cannot: regions with no real cloud climatology at all (most of this
   archive's other 16 regions), and a controllable occlusion curriculum
   (exact coverage fraction, opaque vs. semi-transparent) for the
   degradation curve's fine-grained points. Real and synthetic cover
   different parts of the problem; neither replaces the other everywhere.

## 8. Target volume and composition — is more tiles the answer?

**No, not by itself.** `pipeline.global_a5.yaml`'s own estimate is "roughly
13,600 tiles and about 40 GB" from 1,200 new AOIs — but it uses the **same
clear-anchor selection** as the current archive (`scene_selection:
nearest_clear_mosaic`, ≥98% AOI clarity required to qualify a date). Run as
configured, it adds ~13,600 tiles that are just as cloud-free as today's
95.5%, possibly more so given the tighter 98% filter. Combined
(6,406 + 13,600 ≈ 20,000 tiles), the cloud-occluded fraction stays roughly
4–5% either way — **volume does not fix composition.** For general LULC
accuracy, ~20,000 tiles with pretrained backbones is respectable (in the
range of established benchmarks like LoveDA ~6,000 patches or DFC2020
~6,000). For the cloud/SAR research question specifically, no tile count
fixes it without deliberately changing what fraction is cloudy, multi-date,
or orbit-annotated.

### Cloud-mask bank (real shapes for synthetic injection)

Currently ~277 real cloudy tiles archive-wide = ~277 distinct cloud shapes,
which will visibly repeat across a 60-epoch run. **Target: 500+ distinct real
cloud masks** — harvest from Task 3's freshly-pulled cloudy scenes to get
there.

### Real cloud tiles (Task 3), by purpose

| Purpose | Minimum | Recommended |
|---|---|---|
| Defensible degradation curve (5 buckets: 0/25/50/75/100% occlusion) | ~50–75 tiles/bucket → **~250–300 total** | 150–300/bucket → **~600–1,500 total** |
| Real training signal for cloud-favorable regions (not just validation) | — | **1,000–3,000+**, concentrated in Ghana/Papua/Vietnam/Cuba/Egypt/NE-zone |

Below ~250 total real-cloud tiles, per-bucket averages can't distinguish a
real degradation trend from sampling luck. Below ~50/bucket, one unusual
tile (e.g. a cloud sitting exactly over a rare class) swings the whole
bucket's number.

### Cloud-free vs. cloudy ratio in the raw archive

Keep clear tiles the majority — **70–85% of total volume clear**, 15–30%
carrying meaningful cloud (real or synthetic-eligible). This is consistent
with, not contradicted by, the report's cited literature figure of injecting
cloud into 40–60% of training *batches* (Table 12) — that injection happens
on-the-fly over clear tiles at train time, so the raw archive doesn't need to
*be* 40–60% cloudy; it needs enough distinct clear tiles that repeated
synthetic injection over them doesn't get repetitive.

### Per-class pixel floor — the constraint "just add more tiles" can hide

The two rarest classes, `flooded_veg` (0.55% of pixels, 453,094 px in the
current test split, IoU 0.477) and `snow_ice` (2.1%, 582,296 px, IoU 0.717),
already work reasonably today. **Risk:** if the new global_a5 AOIs skew
toward biomes without wetlands or high-altitude terrain, both classes get
*relatively* rarer even as total tile count triples. Count derivable
per-class pixels in the *planned* AOI list before running the acquisition,
not after — region selection should preserve or grow wetland/high-altitude
representation, not just chase total count.

### Target composition summary

| Component | Current | Add | Target total |
|---|---|---|---|
| Clear/baseline tiles (any region) | 6,406 | +13,600 (global_a5, already planned) | ~20,000 |
| Real cloud-favorable tiles | ~277 scattered | +1,000–3,000 concentrated | ~1,300–3,300 |
| Distinct real cloud-mask shapes | ~277 | harvested from the above | 500+ |
| Multi-temporal SAR scenes | 6,406 (1/tile) | +400–600 (Task 1) | ~6,800–7,000 |
| Tiles with incidence angle | 0 | +all existing (Task 2) + new | 100% of archive |

At ~20,000 tiles / ~1,700+ locations, a 10% val / 10% test split yields
~2,000 tiles each — roughly 3x today's 672/703 — which meaningfully tightens
confidence intervals on every held-out metric reported.

## 9. Source

Full diagnosis, code defects, and the six-phase remediation plan this spec's
Phase 5 comes from: `AETHER_SAR_Cloud_Remediation_Report.docx` in this repo
(§3 for the dataset diagnosis tables, §6 Phase 5, §9 risk register — in
particular the noted risk that single-date SAR may cap SAR-only accuracy near
65–70% even after Phase 1–3 land, making Task 1 "the only remaining lever").
