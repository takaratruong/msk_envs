#!/usr/bin/env python3
"""Render this read-only dashboard without interrupting its long videos.

The shared renderer is unchanged. This project removes its 30-second document
refresh and starts the selected muted documentary when the page is opened.
Rerun this command and refresh Firefox when publishing a new progress state.
"""
import argparse
import html as html_module
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    renderer = Path.home() / '.agentic-workflow/tools/render_dashboard.py'
    subprocess.run([sys.executable, str(renderer), '--project-root', str(root)], check=True)
    page = root / 'dashboard/index.html'
    html = page.read_text()
    refresh = '  <meta http-equiv="refresh" content="30">\n'
    state = json.loads((root/'.workflow/current.json').read_text())
    selected = next(m for m in state.get('media', [])
                    if m.get('type') == 'video' and not m.get('superseded', False))
    source = html_module.escape('../'+selected['path'], quote=True)
    target = f'<video controls preload="metadata" playsinline><source src="{source}">'
    if html.count(refresh) != 1 or html.count(target) != 1:
        raise ValueError('Unexpected dashboard template; refusing an unverified rewrite')
    html = html.replace(refresh, '').replace(target, target.replace(' playsinline>', ' playsinline autoplay muted>'))
    # Give the selected motion enough room to inspect the complete body and
    # hand inset. Historical media keeps the shared compact gallery layout.
    if html.count('</style>') != 1:
        raise ValueError('Unexpected dashboard style block')
    html = html.replace('</style>', '  .media:has(video[autoplay]) { grid-column: 1 / -1; }\n</style>')
    temporary = page.with_suffix('.html.tmp')
    temporary.write_text(html)
    temporary.replace(page)


if __name__ == '__main__':
    main()
