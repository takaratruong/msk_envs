#!/usr/bin/env python3
"""Create a new immutable split without changing source frames or old results."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import sqlite3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();parent=a.parent.resolve();out=a.output.resolve()
    if out.exists():raise RuntimeError('Use a new output directory.')
    out.mkdir(parents=True)
    frames=json.loads((parent/'frames.json').read_text())
    frames['parent_frames_sha256']=hashlib.sha256((parent/'frames.json').read_bytes()).hexdigest()
    frames['parent_run']=str(parent)
    frames['holdout_rule']='interleaved bin%10=5 plus [48,49),[70,71) seconds; train excludes [47.5,49.5),[69.5,71.5)'
    for item in frames['frames']:
        t=item['time_seconds'];split=item['split'];group='interleaved' if split=='val' else None
        if 47.5<=t<49.5 or 69.5<=t<71.5:
            if 48<=t<49 or 70<=t<71:split='val';group='buffered_block'
            else:split='buffer';group=None
        folder={'train':'images','val':'holdout','buffer':'buffer'}[split]
        (out/folder).mkdir(exist_ok=True)
        target=out/folder/item['name']
        os.link(item['image_path'],target)
        item.update(split=split,image_path=str(target),validation_group=group)
    (out/'frames.json').write_text(json.dumps(frames,indent=2)+'\n')
    with sqlite3.connect(parent/'database.db') as src,sqlite3.connect(out/'database.db') as dst:
        src.backup(dst)
    # Cached features/matches can be reused, but only named training observations
    # enter incremental mapping. No camera solution or 3D points are copied.
    (out/'source_masks').symlink_to(parent/'source_masks',target_is_directory=True)
    (out/'database_cache_provenance.json').write_text(json.dumps(dict(
        source_sha256=frames['source_sha256'],parent_database=str(parent/'database.db'),
        copied_database_sha256=hashlib.sha256((out/'database.db').read_bytes()).hexdigest(),
        note='Cached local SIFT descriptors and pair verification only; no poses/3D points. Mapping restricted by image_names.',
        training_names=[x['name'] for x in frames['frames'] if x['split']=='train']),indent=2)+'\n')
    print({split:sum(x['split']==split for x in frames['frames']) for split in ['train','val','buffer']})


if __name__=='__main__':main()
