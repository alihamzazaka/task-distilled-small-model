# Image prompts — Project 02 — Task-Distilled Small Model (invoice → JSON)

> Distil a local 14B teacher into a 0.5B invoice-extraction student. Images: the cost problem, teacher labelling, synthetic-data + schema filter, the distillation transfer, the data-beats-size lever, and the local result.

**How to use:** paste any prompt below into Claude (or Midjourney / DALL·E / Ideogram). Each already ends with the shared style suffix so all images across the six projects read as one series. Generate at 1792×1024 (or 1024×1024 for a square variant). If the tool adds text anyway, append *"absolutely no typography of any kind"*.

These illustrate **how the problem is solved** — problem → method stages → result — not just a hero shot.

---

## Image 1 — The problem: expensive, remote extraction
*Suggested placement: hero / section 1*

> A paper invoice being fed into a distant, oversized cloud-mounted engine that returns a neat structured data-cube, but the connection between them is a long thin metered pipe with a coin-meter ticking and a small padlock indicating data leaving the premises. Convey cost, distance, and egress as the pain. — clean modern technical illustration, flat vector style, isometric where it helps, white background, restrained palette of deep teal (#1D9E75), coral (#D85A30) and slate gray (#5F5E5A) with a single blue accent, subtle depth, no gradients-as-decoration, crisp thin outlines, no text, no words, no letters, no numbers, no logos, no UI chrome, highly detailed, professional documentation / research-paper figure aesthetic, 1792x1024

## Image 2 — The teacher: a large local model labels invoices
*Suggested placement: section 3 / method*

> An isometric large translucent 14-billion-parameter model, depicted as a big multi-layered glowing brain-slab sitting on a local desk (not in a cloud), reading a stack of paper invoices on its left and emitting, on its right, pristine structured JSON-like data blocks locking into a rigid schema frame. A small validation gate checks each block for shape. — clean modern technical illustration, flat vector style, isometric where it helps, white background, restrained palette of deep teal (#1D9E75), coral (#D85A30) and slate gray (#5F5E5A) with a single blue accent, subtle depth, no gradients-as-decoration, crisp thin outlines, no text, no words, no letters, no numbers, no logos, no UI chrome, highly detailed, professional documentation / research-paper figure aesthetic, 1792x1024

## Image 3 — Synthetic data + ruthless filtering
*Suggested placement: section 3 / method*

> A generation-and-filter pipeline: on the left a fountain producing many varied synthetic invoice cards of different layouts, flowing through a strict funnel with a schema-shaped aperture that lets only correctly-shaped data blocks pass while malformed ones fall away into a discard bin, and a second 'consistency' sieve removing near-duplicates. Emphasise selectivity — many in, fewer high-quality out. — clean modern technical illustration, flat vector style, isometric where it helps, white background, restrained palette of deep teal (#1D9E75), coral (#D85A30) and slate gray (#5F5E5A) with a single blue accent, subtle depth, no gradients-as-decoration, crisp thin outlines, no text, no words, no letters, no numbers, no logos, no UI chrome, highly detailed, professional documentation / research-paper figure aesthetic, 1792x1024

## Image 4 — The distillation: 14B teacher into 0.5B student
*Suggested placement: section 3 / method / hero*

> A large translucent teacher brain-slab pouring a concentrated bright stream of distilled knowledge downward into a tiny crystalline student model roughly one-tenth its size standing on the same desk. The small student glows as it fills; paired invoice-and-JSON cards line the transfer beam, showing what is being taught. Convey compression of capability into a small footprint. — clean modern technical illustration, flat vector style, isometric where it helps, white background, restrained palette of deep teal (#1D9E75), coral (#D85A30) and slate gray (#5F5E5A) with a single blue accent, subtle depth, no gradients-as-decoration, crisp thin outlines, no text, no words, no letters, no numbers, no logos, no UI chrome, highly detailed, professional documentation / research-paper figure aesthetic, 1792x1024

## Image 5 — Full fine-tune on the GPU
*Suggested placement: section 3 / method*

> The small student model sitting atop a single glowing GPU die, its every layer lit and updating (a full fine-tune, not just adapters), with training batches of invoice→JSON pairs streaming through it in a tight loop and a loss-curve motif descending in the background as thin descending arcs. — clean modern technical illustration, flat vector style, isometric where it helps, white background, restrained palette of deep teal (#1D9E75), coral (#D85A30) and slate gray (#5F5E5A) with a single blue accent, subtle depth, no gradients-as-decoration, crisp thin outlines, no text, no words, no letters, no numbers, no logos, no UI chrome, highly detailed, professional documentation / research-paper figure aesthetic, 1792x1024

## Image 6 — The lever: data beats size at small scale
*Suggested placement: section 5 / concept*

> Two staircases climbing toward the same high finish flag. One is short-stepped but built from MANY small data-cube blocks and nearly reaches the top; the other is built from a FEW giant model blocks but stops well short. A small determined student robot happily climbs the many-small-steps staircase. Convey that more data, not a bigger model, closed the gap. — clean modern technical illustration, flat vector style, isometric where it helps, white background, restrained palette of deep teal (#1D9E75), coral (#D85A30) and slate gray (#5F5E5A) with a single blue accent, subtle depth, no gradients-as-decoration, crisp thin outlines, no text, no words, no letters, no numbers, no logos, no UI chrome, highly detailed, professional documentation / research-paper figure aesthetic, 1792x1024

## Image 7 — The result: a tiny private model, structured output
*Suggested placement: section 5 / result*

> A compact palm-sized glowing model chip running entirely on a local desk with no cloud pipe attached, cleanly converting a paper invoice into a neat locked structured-data record, a small teal check-mark confirming schema validity and a shield motif indicating the data never left the desk. — clean modern technical illustration, flat vector style, isometric where it helps, white background, restrained palette of deep teal (#1D9E75), coral (#D85A30) and slate gray (#5F5E5A) with a single blue accent, subtle depth, no gradients-as-decoration, crisp thin outlines, no text, no words, no letters, no numbers, no logos, no UI chrome, highly detailed, professional documentation / research-paper figure aesthetic, 1792x1024

---

### Consistency tips
- Keep the same tool, seed, and style across all 7 so they match.
- The palette (deep teal / coral / slate + one blue accent) is shared with the other five projects and with the generated flow figures (`figures/fig_*.png`).
- These are the *artistic* explainer images; the quantitative problem→solution flow figures are already generated programmatically in `figures/`.
