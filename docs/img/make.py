"""Render the X article images in the hotin.ai chart style: HTML per image -> headless Chrome PNG (2x).
Numbers are from benchmark/ results."""
import os, subprocess, tempfile

OUT = os.path.dirname(os.path.abspath(__file__))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# hotin.ai tokens (light only): heat accent = "ours", everything else muted gray.
ACCENT, INK, MUTED, BAR, BG, LINE_SOFT = "#da3200", "#151b24", "#4f5661", "#aab0b8", "#f7f9fc", "#e5e8ec"
FONT_DISPLAY = "'Geist', -apple-system, 'Helvetica Neue', Arial, sans-serif"
FONT_MONO = "'Geist Mono', ui-monospace, Menlo, Consolas, monospace"

BASE = f"""<!doctype html><meta charset=utf-8>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;600;650;700&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0}} body{{background:{BG};color:{INK};font-family:{FONT_DISPLAY};padding:56px 64px}}
h1{{font-size:44px;font-weight:700;letter-spacing:-.5px}} .sub{{font-size:24px;color:{MUTED};margin-top:8px}}
.src{{position:absolute;bottom:28px;left:64px;font-size:15px;color:{MUTED};font-family:{FONT_MONO}}}
</style>"""


def bars(title, rows, maxv, unit="%", width=560, lw=250):
    """rows: (label, value, accent?, note). One horizontal bar per row on a horizontal gridline,
    value labeled at the end in mono (hotin chart recipe: one heat-colored bar is "ours", the rest muted gray)."""
    h = f'<div style="font-size:28px;font-weight:650;margin:0 0 18px">{title}</div>'
    for label, v, acc, note in rows:
        w = max(4, v / maxv * width)
        h += (f'<div style="display:flex;align-items:center;height:54px;margin-bottom:10px;'
              f'border-bottom:1px solid {LINE_SOFT}">'
              f'<div style="width:{lw}px;flex-shrink:0;font-size:21px;color:{INK if acc else MUTED};font-weight:{650 if acc else 450}">{label}</div>'
              f'<div style="width:{w}px;height:34px;background:{ACCENT if acc else BAR};border-radius:0 4px 4px 0"></div>'
              f'<div style="margin-left:14px;font-size:22px;font-weight:650;font-family:{FONT_MONO}">{v:g}{unit}</div>'
              f'<div style="margin-left:10px;font-size:16px;color:{MUTED};white-space:nowrap;font-family:{FONT_MONO}">{note}</div></div>')
    return h


PAGES = {
    "1-cover": ((1500, 600), f"""
<div style="height:100%;display:flex;flex-direction:column;justify-content:center">
<div style="font-size:24px;color:{MUTED};font-weight:600;letter-spacing:1px">CLM-8B · APPLE SILICON · MLX</div>
<h1 style="font-size:64px;margin-top:14px;line-height:1.08">Out of the box it lost.<br>Trained on a Mac, it learned.</h1>
<div style="display:flex;gap:56px;margin-top:40px">
 <div><div style="font-size:56px;font-weight:750;color:{ACCENT}">22% → 50%</div><div class=sub>web actions, ~40 s of training</div></div>
 <div><div style="font-size:56px;font-weight:750;color:{ACCENT}">3.6% → 86%</div><div class=sub>77 support routes</div></div>
</div></div>"""),

    "2-how-clm-works": ((1600, 900), f"""
<h1>How CLM makes a decision</h1><div class=sub>The 8B encoder is frozen. Only the two small heads are trained. (Example values.)</div>
<svg width="1470" height="640" viewBox="0 0 1470 640" style="margin-top:24px" font-family="-apple-system,Helvetica,Arial">
<defs><marker id=a viewBox="0 0 10 10" refX=9 refY=5 markerWidth=8 markerHeight=8 orient=auto><path d="M0,0L10,5L0,10z" fill="{MUTED}"/></marker></defs>
<g font-size=22>
 <rect x=0 y=40 width=330 height=150 rx=14 fill="#F2F4F7"/><text x=24 y=82 font-weight=650>Situation</text>
 <text x=24 y=120 fill="{MUTED}" font-size=19>"Checkout page, cart has</text><text x=24 y=146 fill="{MUTED}" font-size=19>2 items, user wants to pay"</text>
 <rect x=0 y=330 width=330 height=270 rx=14 fill="#F2F4F7"/><text x=24 y=372 font-weight=650>Options, any text</text>
 <text x=24 y=412 fill="{MUTED}" font-size=19>[button] Place order</text><text x=24 y=444 fill="{MUTED}" font-size=19>[link] Edit cart</text>
 <text x=24 y=476 fill="{MUTED}" font-size=19>[button] Apply coupon</text><text x=24 y=508 fill="{MUTED}" font-size=19>…different on every page</text>
 <rect x=440 y=150 width=300 height=340 rx=14 fill="#E3E6EB"/><text x=590 y=300 text-anchor=middle font-weight=700 font-size=26>Qwen3-8B</text>
 <text x=590 y=334 text-anchor=middle fill="{MUTED}">encoder, frozen</text><text x=590 y=364 text-anchor=middle fill="{MUTED}" font-size=19>(options cached)</text>
 <rect x=850 y=60 width=260 height=110 rx=14 fill="{ACCENT}"/><text x=980 y=108 text-anchor=middle fill="#fff" font-weight=700>State head</text>
 <text x=980 y=140 text-anchor=middle fill="#fff" font-size=18>trainable</text>
 <rect x=850 y=410 width=260 height=110 rx=14 fill="{ACCENT}"/><text x=980 y=458 text-anchor=middle fill="#fff" font-weight=700>Action head</text>
 <text x=980 y=490 text-anchor=middle fill="#fff" font-size=18>trainable</text>
 <rect x=1200 y=210 width=270 height=210 rx=14 fill="#F2F4F7"/><text x=1335 y=262 text-anchor=middle font-weight=650>Similarity</text>
 <text x=1335 y=298 text-anchor=middle fill="{MUTED}" font-size=19>score every option</text>
 <text x=1335 y=352 text-anchor=middle font-weight=700 fill="{ACCENT}">Place order 0.81</text>
 <text x=1335 y=386 text-anchor=middle fill="{MUTED}" font-size=19>Edit cart 0.12 · …</text>
</g>
<g stroke="{MUTED}" stroke-width=3 fill=none marker-end="url(#a)">
 <path d="M330,115 L440,230"/><path d="M330,465 L440,410"/>
 <path d="M740,230 L850,115"/><path d="M740,410 L850,465"/>
 <path d="M1110,115 L1200,260"/><path d="M1110,465 L1200,370"/></g>
<text x=980 y=620 text-anchor=middle font-size=20 fill="{MUTED}">Heads train in ~40 s on a MacBook</text>
</svg>"""),

    "3-trained-vs-not": ((1600, 760), f"""
<h1>Training the heads is the whole story</h1><div class=sub>Held-out test sets. Only CLM was trained; laya-mlx ran out of the box.</div>
<div style="display:flex;gap:70px;margin-top:40px"><div>{bars("Web: pick 1 of 15 page elements (Mind2Web)", [
    ("CLM trained", 50.0, True, ""), ("laya-mlx", 40.7, False, ""),
    ("BM25 keywords", 28.0, False, ""), ("CLM out of the box", 22.0, False, "")], 100, width=380)}</div>
<div>{bars("Routing: 77 support routes (Banking77)", [
    ("CLM trained", 85.6, True, ""), ("laya-mlx", 40.1, False, ""),
    ("BM25 keywords", 33.7, False, ""), ("CLM out of the box", 3.6, False, "")], 100, width=380)}</div></div>
<div class=src>CLM trained = mean of 3 seeds (web 49.3 to 51.3%, routing 84.2 to 86.4%), on the Mac from MLX encodings. Mind2Web split by task: 1,306 training steps, 150 test steps.</div>"""),

    "4-unseen-routes": ((1600, 640), f"""
<h1>Where CLM beats a plain classifier</h1><div class=sub>Trained on 60 routes, tested on 17 routes it never saw, each described in one line of text.</div>
<div style="margin-top:50px">{bars("Accuracy among the 17 new routes", [
    ("CLM trained", 50.1, True, ""), ("CLM released head", 16.5, False, ""),
    ("Classifier", 0, False, "can't output a label it never trained on")], 100, width=700)}</div>
<div class=src>On the 77 known routes a classifier on the same encodings is as good (87.2% vs 85.6%). For fixed labels, use one.</div>"""),

    "5-head-to-head": ((1600, 640), f"""
<h1>Three MLX ports, same Mac</h1><div class=sub>Median latency, ms (lower is better). 3 interleaved rounds, run one at a time. M5 Pro.</div>
<div style="display:flex;gap:44px;margin-top:40px">
<div>{bars("One decision", [("clm-tune-mlx", 83, True, ""), ("RealityCat", 97, False, ""), ("czl 8-bit", 100, False, "")], 240, " ms", 200, 160)}</div>
<div>{bars("8 questions, one state", [("clm-tune-mlx", 190, True, ""), ("RealityCat", 201, False, ""), ("czl 8-bit", 226, False, "")], 240, " ms", 200, 160)}</div>
<div>{bars("589-option menu", [("clm-tune-mlx", 110, True, ""), ("RealityCat", 120, False, ""), ("czl 8-bit", 133, False, "")], 240, " ms", 200, 160)}</div></div>
<div class=src>Fastest in 10 of 12 paired timings. Same answers every round. RealityCat uses the least memory (8.4 vs 9.0 GB).</div>"""),
}

tmp = tempfile.mkdtemp()
for name, ((w, h), body) in PAGES.items():
    html = os.path.join(tmp, name + ".html")
    open(html, "w").write(BASE + f"<body style='width:{w}px;height:{h}px;position:relative'>{body}</body>")
    png = os.path.join(OUT, name + ".png")
    if os.path.exists(png):
        os.remove(png)
    try:  # ponytail: headless Chrome writes the PNG but often never exits; the file is the success signal
        subprocess.run([CHROME, "--headless=new", f"--user-data-dir={tmp}/prof", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--window-size={w},{h}", f"--screenshot={png}", "file://" + html], capture_output=True, timeout=25)
    except subprocess.TimeoutExpired:
        pass
    if not os.path.exists(png):
        raise SystemExit(f"no screenshot for {name}")
    print(png)
