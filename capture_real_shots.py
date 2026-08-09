#!/usr/bin/env python3
"""Capture real UI screenshots from Douban / Hupu on a connected phone (adb).

These are primarily hard negatives for YOLO skip-button training (normal app
UI with no splash-ad skip control). Output: real_shots/raw/<app>_<nnn>.png

Usage:
  python capture_real_shots.py              # both apps, ~20 shots each
  python capture_real_shots.py hupu 30
  python capture_real_shots.py douban 15
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "real_shots", "raw")
ADB = os.environ.get("ADB", "adb")

APPS = {
    "hupu": {
        "pkg": "com.hupu.games",
        "activity": "com.hupu.games/.main.MainActivity",
        # bottom-tab labels to visit (partial match)
        "tabs": ["首页", "专区", "深聊", "我的"],
        "feed_tabs": ["热榜", "推荐", "NBA", "国际足球"],
    },
    "douban": {
        "pkg": "com.douban.frodo",
        "activity": "com.douban.frodo/.activity.SplashActivity",
        "tabs": ["书影音", "小组", "市集", "我"],  # may vary by version
        "feed_tabs": ["动态", "推荐", "豆瓣热门", "电影", "读书"],
    },
}


def sh(*args: str, check: bool = True, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [ADB, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def shell(cmd: str, **kw) -> subprocess.CompletedProcess:
    return sh("shell", cmd, **kw)


def sleep(s: float = 1.0) -> None:
    time.sleep(s)


def screencap(path: str) -> None:
    # Prefer exec-out binary PNG to avoid device-side path quirks
    data = subprocess.check_output([ADB, "exec-out", "screencap", "-p"], timeout=30)
    # Some devices insert CRLF; normalize if needed
    if data[:8] != b"\x89PNG\r\n\x1a\n" and b"\r\n" in data[:200]:
        data = data.replace(b"\r\n", b"\n")
    with open(path, "wb") as f:
        f.write(data)


def dump_ui() -> ET.Element | None:
    try:
        shell("uiautomator dump /sdcard/uidump.xml", check=False)
        raw = subprocess.check_output(
            [ADB, "exec-out", "cat", "/sdcard/uidump.xml"], timeout=30
        )
        return ET.fromstring(raw)
    except Exception as e:
        print(f"  [ui dump failed] {e}", flush=True)
        return None


def parse_bounds(b: str) -> tuple[int, int, int, int] | None:
    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", b or "")
    if not m:
        return None
    return tuple(int(x) for x in m.groups())  # type: ignore[return-value]


def nodes_with_text(root: ET.Element) -> list[dict]:
    out = []
    for n in root.iter("node"):
        text = (n.get("text") or "").strip()
        desc = (n.get("content-desc") or "").strip()
        label = text or desc
        if not label:
            continue
        bounds = parse_bounds(n.get("bounds") or "")
        if not bounds:
            continue
        out.append(
            {
                "label": label,
                "clickable": n.get("clickable") == "true",
                "bounds": bounds,
                "cx": (bounds[0] + bounds[2]) // 2,
                "cy": (bounds[1] + bounds[3]) // 2,
            }
        )
    return out


def find_label(items: list[dict], needle: str, y_min: int = 0, y_max: int = 10_000) -> dict | None:
    for it in items:
        if needle in it["label"] and y_min <= it["cy"] <= y_max:
            return it
    return None


def tap(x: int, y: int) -> None:
    shell(f"input tap {x} {y}", check=False)


def swipe_up(duration_ms: int = 350) -> None:
    # Portrait 1080x2400-ish: scroll feed
    shell(f"input swipe 540 1700 540 700 {duration_ms}", check=False)


def swipe_left(duration_ms: int = 250) -> None:
    shell(f"input swipe 900 1200 200 1200 {duration_ms}", check=False)


def press_back() -> None:
    shell("input keyevent 4", check=False)


def force_stop(pkg: str) -> None:
    shell(f"am force-stop {pkg}", check=False)


def launch(activity: str) -> None:
    shell(f"am start -n {activity}", check=False)


def dismiss_common_overlays(root: ET.Element | None) -> bool:
    """Tap common splash/permission dismiss buttons if present. Returns True if tapped."""
    if root is None:
        return False
    items = nodes_with_text(root)
    for needle in (
        "跳过",
        "跳过广告",
        "关闭",
        "我知道了",
        "知道了",
        "同意",
        "允许",
        "以后再说",
        "暂不",
        "取消",
        "关闭广告",
        "略过",
    ):
        it = find_label(items, needle)
        if it and it["clickable"]:
            print(f"  dismiss: {it['label']!r} @ ({it['cx']},{it['cy']})", flush=True)
            tap(it["cx"], it["cy"])
            sleep(0.8)
            return True
    return False


def next_path(app: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    existing = [
        f
        for f in os.listdir(OUT)
        if f.startswith(f"{app}_") and f.endswith(".png")
    ]
    n = 0
    if existing:
        nums = []
        for f in existing:
            m = re.search(rf"{re.escape(app)}_(\d+)", f)
            if m:
                nums.append(int(m.group(1)))
        n = (max(nums) + 1) if nums else len(existing)
    return os.path.join(OUT, f"{app}_{n:03d}.png")


def shot(app: str, note: str = "") -> str:
    path = next_path(app)
    screencap(path)
    print(f"  saved {os.path.basename(path)}  {note}", flush=True)
    return path


def explore_app(app: str, n_target: int) -> int:
    cfg = APPS[app]
    print(f"\n=== {app} ({cfg['pkg']}) target={n_target} ===", flush=True)
    force_stop(cfg["pkg"])
    sleep(0.5)
    launch(cfg["activity"])
    sleep(2.5)

    # Splash / permission dance + capture anything that looks like ads
    for _ in range(6):
        root = dump_ui()
        shot(app, "post-launch")
        if not dismiss_common_overlays(root):
            break
        sleep(0.5)

    count_start = len([f for f in os.listdir(OUT) if f.startswith(f"{app}_")])

    # Visit feed sub-tabs near the top
    root = dump_ui()
    items = nodes_with_text(root) if root is not None else []
    for tab in cfg.get("feed_tabs", []):
        it = find_label(items, tab, y_max=500)
        if it:
            tap(it["cx"], it["cy"])
            sleep(1.2)
            shot(app, f"feed-tab:{tab}")
            swipe_up()
            sleep(0.8)
            shot(app, f"feed-tab:{tab}:scrolled")
            root2 = dump_ui()
            items = nodes_with_text(root2 if root2 is not None else ET.Element("hierarchy"))

    # Scroll main feed several times
    for i in range(6):
        swipe_up()
        sleep(0.7)
        shot(app, f"scroll:{i}")
        # occasionally open a mid-screen post
        if i % 2 == 0:
            tap(540, 1100)
            sleep(1.5)
            shot(app, f"detail:{i}")
            # light scroll inside detail
            swipe_up()
            sleep(0.6)
            shot(app, f"detail-scroll:{i}")
            press_back()
            sleep(0.8)

    # Bottom navigation tabs
    root = dump_ui()
    items = nodes_with_text(root) if root is not None else []
    for tab in cfg.get("tabs", []):
        it = find_label(items, tab, y_min=2100)
        if not it:
            # try without y filter
            it = find_label(items, tab)
        if not it:
            print(f"  tab not found: {tab}", flush=True)
            continue
        tap(it["cx"], it["cy"])
        sleep(1.3)
        shot(app, f"tab:{tab}")
        for j in range(2):
            swipe_up()
            sleep(0.6)
            shot(app, f"tab:{tab}:scroll{j}")
        # refresh items after tab switch
        root = dump_ui()
        items = nodes_with_text(root) if root is not None else []

    # Horizontal swipes on home (story/carousels)
    for k in range(3):
        # ensure roughly on first tab
        if items:
            home = find_label(items, cfg["tabs"][0] if cfg["tabs"] else "首页", y_min=2100)
            if home:
                tap(home["cx"], home["cy"])
                sleep(0.8)
        swipe_left()
        sleep(0.7)
        shot(app, f"hswipe:{k}")

    count_end = len([f for f in os.listdir(OUT) if f.startswith(f"{app}_")])
    got = count_end - count_start
    # If short of target, pure scroll capture
    while got < n_target:
        swipe_up()
        sleep(0.6)
        shot(app, "fill")
        got += 1
        if got >= n_target + 5:  # safety
            break

    print(f"=== {app} captured +{got} (total {count_end}) ===", flush=True)
    return got


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    apps = []
    n = 20
    for a in args:
        if a.isdigit():
            n = int(a)
        elif a in APPS:
            apps.append(a)
        elif a in ("both", "all"):
            apps = list(APPS)
        else:
            print(f"unknown arg: {a}", file=sys.stderr)
            return 2
    if not apps:
        apps = list(APPS)

    # device check
    try:
        r = sh("devices")
    except FileNotFoundError:
        print("adb not found; set ADB= or install platform-tools", file=sys.stderr)
        return 1
    if "\tdevice" not in r.stdout:
        print("no adb device in 'device' state:\n" + r.stdout, file=sys.stderr)
        return 1

    os.makedirs(OUT, exist_ok=True)
    total = 0
    for app in apps:
        total += explore_app(app, n)
    print(f"\nDONE: {total} new shots under {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
