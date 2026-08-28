#!/usr/bin/env python3
"""
scripts/make_event_gifs.py

Generate annotated GIFs from NDJSON gameplay logs and frames/mp4. This script
matches the visual style of the repository GIFs and draws labels for events
Repeat / Seam / Plus and a circle marker for PLUS using TikZ/ztoc-style iMap
tuples when available (iMap, imap, i_map). It prefers iMap coordinates over x/y
when present.

Usage examples (run from repo root):

1) Basic usage (frames already extracted):
   python3 scripts/make_event_gifs.py \
     --ndjson data/records.ndjson \
     --frames-dir frames \
     --frame-pattern "out_{step:06d}.png" \
     --out gifs --ref-gif 2706plus32kstep7122389seed.gif --fps 10

2) If you only have an mp4 and need to extract frames first (10 FPS):
   mkdir -p frames
   ffmpeg -i mp4demoParameterGolfTim.mp4 -vf "fps=10,scale=640:-1" frames/out_%06d.png
   python3 scripts/make_event_gifs.py --ndjson data/records.ndjson --frames-dir frames --out gifs --ref-gif 2706plus32kstep7122389seed.gif

3) Control mapping of iMap / x,y -> pixels (TikZ ztoc style):
   --xy-scale S    multiply iMap/x,y units by S before projecting (default 8.0)
   --xy-offset X Y add pixel offset after scaling (useful to center)
   Use --xy-scale 1 if coordinates are already in pixels.

Notes about iMap/TikZ/ztoc mapping:
  - The script checks for iMap variants: list/tuple like [i,j], string "(i,j)", or dict {"i":...,"j":...}.
  - Mapping rule (ztoc/TikZ): image center corresponds to (0,0); positive y is up.
    px = image_width/2 + i * xy_scale + xy_offset_x
    py = image_height/2 - j * xy_scale + xy_offset_y

Requirements:
  pip install pillow imageio tqdm
  ffmpeg (optional, used for palette matching if available)

Outputs:
  gifs/<run_id>.gif  (one annotated GIF per run_id found in NDJSON)

"""

import os
import sys
import json
import argparse
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Any

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    print("Please install pillow: pip install pillow")
    raise

try:
    import imageio
except Exception:
    imageio = None

from tqdm import tqdm

EVENT_TARGETS = {"REPEAT", "SEAM", "PLUS", "Repeat", "Seam", "Plus"}
DEFAULT_FRAME_PATTERN = "out_{step:06d}.png"

# default action colors (RGBA)
COLORS = {
    'PLUS': (255, 50, 50, 220),
    'SEAM': (0, 200, 255, 220),
    'REPEAT': (255, 200, 0, 220),
    'OTHER': (200, 200, 200, 180)
}


def load_ndjson(path: str):
    runs = defaultdict(list)
    p = Path(path)
    if p.is_dir():
        files = list(p.glob('*.ndjson'))
    else:
        files = [p]
    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # skip malformed lines
                    continue
                run_id = obj.get('run_id') or obj.get('run') or obj.get('runId') or 'run_unknown'
                runs[run_id].append(obj)
    return runs


def frame_path(frames_dir: str, pattern: str, step: int) -> str:
    fname = pattern.format(step=int(step))
    if os.path.isabs(fname):
        return fname
    return os.path.join(frames_dir, fname)


def ensure_font(size=18):
    # try common fonts, fallback to default
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("Arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def parse_imap(ev: dict) -> Optional[Tuple[float, float]]:
    """
    Extract iMap/i_map/imap coordinate pairs in several formats.
    Accepts:
      - list/tuple: [i, j]
      - string: "(i,j)" or "i,j"
      - dict: {"i": i, "j": j} or {"x":..,"y":..}
    Returns (i, j) as floats or None.
    """
    # common keys
    for key in ('iMap', 'imap', 'i_map', 'i-map'):
        if key in ev:
            v = ev[key]
            return _parse_pair_like(v)
    # also accept 'imap_tuple' or 'ztoc' if present
    for key in ('imap_tuple', 'ztoc', 'ztoc_tuple'):
        if key in ev:
            return _parse_pair_like(ev[key])
    # fallback to x/y where present
    if 'x' in ev and 'y' in ev:
        try:
            return float(ev['x']), float(ev['y'])
        except Exception:
            pass
    # sometimes embedded in action details or other string fields; try basic search
    for k, v in ev.items():
        if isinstance(v, str) and ('(' in v and ',' in v and ')' in v):
            # try to extract first pair
            try:
                s = v[v.index('(')+1:v.index(')')]
                a,b = s.split(',')[:2]
                return float(a), float(b)
            except Exception:
                continue
    return None


def _parse_pair_like(v: Any) -> Optional[Tuple[float, float]]:
    try:
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        if isinstance(v, dict):
            # accept keys i/j or x/y
            if 'i' in v and 'j' in v:
                return float(v['i']), float(v['j'])
            if 'x' in v and 'y' in v:
                return float(v['x']), float(v['y'])
        if isinstance(v, str):
            s = v.strip()
            # formats: "(i,j)" or "i,j"
            if s.startswith('(') and s.endswith(')'):
                s = s[1:-1]
            if ',' in s:
                a,b = s.split(',')[:2]
                return float(a), float(b)
    except Exception:
        return None
    return None


def annotate_frame(img_path: str, events_at_step: list, xy_scale: float, xy_off: Tuple[float, float]):
    im = Image.open(img_path).convert('RGBA')
    draw = ImageDraw.Draw(im)
    font = ensure_font(18)
    w,h = im.size
    # top-left stack labels
    label_y = 6
    for ev in events_at_step:
        evtype = (ev.get('t') or ev.get('type') or '')
        if not evtype:
            continue
        evtype_up = evtype.upper()
        label = evtype
        color = COLORS.get(evtype_up, COLORS['OTHER'])
        tw, th = draw.textsize(label, font=font)
        draw.rectangle([(6,label_y),(6+tw+8,label_y+th+4)], fill=(0,0,0,140))
        draw.text((10,label_y+2), label, font=font, fill=color)
        label_y += th + 8
        if evtype_up == 'PLUS':
            # Prefer iMap parsing; fallback to x,y
            coords = parse_imap(ev)
            if coords is None:
                # fallback: try keys 'x','y' explicitly
                try:
                    coords = (float(ev.get('x', 0.0)), float(ev.get('y', 0.0)))
                except Exception:
                    coords = None
            if coords is not None:
                i, j = coords
                # ztoc/TikZ style: center(0,0), y positive up -> map to pixels
                px = int(w/2 + i * xy_scale + xy_off[0])
                py = int(h/2 - j * xy_scale + xy_off[1])
                r = max(4, int(6 * max(1.0, xy_scale/8.0)))
                draw.ellipse([(px-r,py-r),(px+r,py+r)], outline=color[:3]+(255,), width=3)
                draw.line([(px-r,py),(px+r,py)], fill=color[:3]+(200,), width=2)
                draw.line([(px,py-r),(px,py+r)], fill=color[:3]+(200,), width=2)
    return im.convert('RGB')


def extract_frames_from_mp4(mp4_path: str, frames_dir: str, fps: int=10, width: Optional[int]=None):
    os.makedirs(frames_dir, exist_ok=True)
    vf = f"fps={fps}"
    if width:
        vf += f",scale={width}:-1"
    cmd = [
        'ffmpeg', '-y', '-i', mp4_path, '-vf', vf, os.path.join(frames_dir, 'out_%06d.png')
    ]
    print('Running ffmpeg to extract frames:', ' '.join(cmd))
    subprocess.check_call(cmd)


def palette_match_gif(input_gif: str, ref_gif: str, out_gif: str):
    # If ffmpeg available, generate palette from ref_gif and apply to input
    try:
        palette = 'palette_from_ref.png'
        cmd1 = ['ffmpeg', '-y', '-i', ref_gif, '-vf', 'palettegen', palette]
        subprocess.check_call(cmd1)
        cmd2 = ['ffmpeg', '-y', '-i', input_gif, '-i', palette, '-lavfi', 'paletteuse', out_gif]
        subprocess.check_call(cmd2)
        os.remove(palette)
        return True
    except Exception:
        return False


def build_gifs(runs: dict, frames_dir: str, pattern: str, out_dir: str, fps: int, ref_gif: Optional[str], xy_scale: float, xy_off: Tuple[float,float], max_frames: Optional[int]):
    os.makedirs(out_dir, exist_ok=True)
    for run_id, events in tqdm(runs.items(), desc='runs'):
        by_step = defaultdict(list)
        all_steps = []
        for ev in events:
            st = ev.get('step')
            if st is None:
                continue
            by_step[int(st)].append(ev)
            all_steps.append(int(st))
        if not all_steps:
            continue
        steps = sorted(set(all_steps))
        frames_imgs = []
        for st in steps:
            if max_frames and len(frames_imgs) >= max_frames:
                break
            fp = frame_path(frames_dir, pattern, st)
            if not os.path.exists(fp):
                # skip silently
                continue
            evs_here = [e for e in by_step[st] if ( (e.get('t') in EVENT_TARGETS) or (e.get('type') in EVENT_TARGETS) )]
            if evs_here:
                img = annotate_frame(fp, evs_here, xy_scale, xy_off)
            else:
                img = Image.open(fp).convert('RGB')
            frames_imgs.append(img)
        if not frames_imgs:
            continue
        temp_out = os.path.join(out_dir, f"{run_id}_temp.gif")
        final_out = os.path.join(out_dir, f"{run_id}.gif")
        # Save using PIL first
        frames_imgs[0].save(temp_out, save_all=True, append_images=frames_imgs[1:], duration=int(1000/fps), loop=0, optimize=False)
        # If ref_gif provided, attempt palette match for consistent style
        if ref_gif and os.path.exists(ref_gif):
            ok = palette_match_gif(temp_out, ref_gif, final_out)
            if ok:
                os.remove(temp_out)
            else:
                # fallback rename
                os.replace(temp_out, final_out)
        else:
            os.replace(temp_out, final_out)
        print(f'Wrote {final_out} ({len(frames_imgs)} frames)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ndjson', required=True, help='NDJSON file or directory')
    p.add_argument('--frames-dir', help='Directory with extracted frames (required unless --mp4 provided)')
    p.add_argument('--mp4', help='Optional mp4 to extract frames from (if frames missing)')
    p.add_argument('--frame-pattern', default=DEFAULT_FRAME_PATTERN)
    p.add_argument('--out', default='gifs')
    p.add_argument('--fps', type=int, default=10)
    p.add_argument('--ref-gif', help='Reference GIF to match palette/size/style (optional)')
    p.add_argument('--xy-scale', type=float, default=8.0, help='Scale factor from iMap/game units to pixels (default heuristic)')
    p.add_argument('--xy-offset', type=float, nargs=2, default=(0.0, 0.0), help='Pixel offset (x_off y_off) after scaling')
    p.add_argument('--max-frames', type=int, default=None, help='Limit frames per GIF (for fast tests)')
    args = p.parse_args()

    if not args.frames_dir and not args.mp4:
        print('Either --frames-dir or --mp4 must be provided')
        sys.exit(1)

    if args.mp4 and not args.frames_dir:
        # create a frames dir next to mp4
        frames_dir = 'frames'
    else:
        frames_dir = args.frames_dir

    if args.mp4 and (not frames_dir or not os.path.exists(frames_dir) or not list(Path(frames_dir).glob('*.png'))):
        print('Extracting frames from mp4...')
        extract_frames_from_mp4(args.mp4, frames_dir, fps=args.fps)

    runs = load_ndjson(args.ndjson)
    build_gifs(runs, frames_dir, args.frame_pattern, args.out, fps=args.fps, ref_gif=args.ref_gif, xy_scale=args.xy_scale, xy_off=tuple(args.xy_offset), max_frames=args.max_frames)

if __name__ == '__main__':
    main()
