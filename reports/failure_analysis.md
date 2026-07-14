# Categorized failure analysis — student vs gold (SPEC deliverable 5)

_37 human-verified gold documents · 711 compared fields (same flatten + tolerance rules as the shipped eval) · **20/37 documents fully correct** · 25 field errors (3.5% of fields). Regenerate: `python scripts/05_failure_analysis.py`._

## Where the residual gap lives (by category)

| Category | Errors | Share | Typical cause / routing guidance |
|---|---:|---:|---|
| `wrong_text` | 11 | 44% | vendor/description normalization drift |
| `hallucinated_line_item` | 4 | 16% | noisy inputs where headers/totals get read as items |
| `wrong_money` | 4 | 16% | amounts beyond the tolerance — usually OCR-ish digit slips or tax/subtotal confusion; a cheap grand-total==sum check catches most |
| `hallucinated_field` | 2 | 8% | student fills optional scalars the gold marks absent |
| `missing_field` | 2 | 8% | optional scalars (payment_terms, tax) absent from sparse inputs |
| `wrong_date` | 2 | 8% | ambiguous or non-ISO source formats (DD/MM vs MM/DD) |

## Per-field breakdown

| Field (leaf) | Errors |
|---|---:|
| `description` | 5 |
| `vendor` | 5 |
| `payment_terms` | 5 |
| `total` | 2 |
| `date` | 2 |
| `qty` | 1 |
| `unit_price` | 1 |
| `tax` | 1 |
| `grand_total` | 1 |
| `subtotal` | 1 |
| `currency` | 1 |

## Worst documents (gold vs pred excerpts)

### `seed-a5cb91c1331c7674` — 5 error(s) / 15 fields

> INVOICE NO: 345678 DATE: 14.03.2024 VENDOR: KAPIL CONSTRUCTION SUPPLIES, DELHI, INDIA CUSTOMER: VIJAY KUMAR, 1…

- **line_items[1].description** (hallucinated_line_item): gold=`None` → pred=`Subtotal`
- **line_items[1].qty** (hallucinated_line_item): gold=`None` → pred=`1.0`
- **line_items[1].total** (hallucinated_line_item): gold=`None` → pred=`0.0`
- **line_items[1].unit_price** (hallucinated_line_item): gold=`None` → pred=`0.0`
- **tax** (wrong_money): gold=`5760.0` → pred=`2880.0`

### `seed-50313abf4bd69246` — 3 error(s) / 15 fields

> Receipt #RC-258974

Vendor: WebCrafters Ltd.
Address: Al-Khwarizmi Street, 78, Riyadh, Saudi Arabia

Client: B…

- **currency** (wrong_text): gold=`SAR` → pred=`ARS`
- **date** (wrong_date): gold=`2024-09-05` → pred=`2024-05-09`
- **vendor** (wrong_text): gold=`WebCrafters Ltd.` → pred=`WebCrafters Ltd`

### `seed-4150f7c8673ecba4` — 2 error(s) / 27 fields

> Store: Costco | 1010 Main St, Vancouver, BC V6B 2Y3 | Tel: 604-444-5555 | Date: 03/17/2024

Item Description  …

- **line_items[2].description** (wrong_text): gold=`Frozen Pizza, Pepperoni 2 pkgs` → pred=`Frozen Pizza, Pepperoni`
- **line_items[3].description** (wrong_text): gold=`Laundry Detergent, Concentrate 30 oz` → pred=`Laundry Detergent, Concentrate`

### `seed-b82cb676efff7d14` — 2 error(s) / 19 fields

> 14.03.2024

Le Jardin du Soleil
12 Rue de la Paix, 75002 Paris, France
Tel: +33 1 23 45 67 89

Facture #FR2024…

- **grand_total** (wrong_money): gold=`10.98` → pred=`12.43`
- **subtotal** (wrong_money): gold=`9.15` → pred=`10.6`

### `seed-cba016eb18af33ee` — 1 error(s) / 19 fields

> Factura de Servicios - 800 Hotel da Ilha, Fortaleza, Brazil

Data: 25/03/2024
Check-in: 22/03/2024 - Check-out…

- **vendor** (wrong_text): gold=`800 Hotel da Ilha` → pred=`Hotel da Ilha`

## When to still call the teacher (the honest routing rule)

The buckets above are the escalation policy: keep the student for short/medium invoices (it is at or near teacher parity there) and route to the teacher when (a) the document has an unusually high line-item count, (b) the cheap arithmetic check `grand_total ≈ Σ line totals + tax` fails on the student's output, or (c) required scalars come back null. Categories, not vibes — each rule maps to a bucket measured above.
