"""Conservative pose ID correction using independently reviewed visible landmarks.

No orientation is inferred from symmetric court geometry alone.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import zipfile

PAIRS = [(0, 5), (25, 27), (1, 6), (2, 7), (26, 28), (3, 8),
         (14, 23), (15, 20), (16, 21), (31, 29), (13, 11)]
# Verified against keypoints-ovj9c's native skeleton: deleted IDs are omitted
# from the version 8 YOLO export, whose arrays use contiguous positions.
ROBOFLOW_ACTIVE_IDS = list(range(17)) + [20, 21, 23, 25, 26, 27, 28, 29, 31]
EXPORT_PAIRS = [(ROBOFLOW_ACTIVE_IDS.index(a), ROBOFLOW_ACTIVE_IDS.index(b)) for a, b in PAIRS]


def pairs_for_export(count):
    return EXPORT_PAIRS if count == 26 else PAIRS


def swap(points, pairs=PAIRS):
    result = copy.deepcopy(points)
    for a, b in pairs:
        result[a], result[b] = result[b], result[a]
    return result


def decide(points, evidence, pairs=PAIRS):
    if any(max(p) >= len(points) for p in pairs):
        return 'review', 'schema_mismatch: requested IDs exceed keypoint count'
    if not evidence or not evidence.get('reviewed') or not evidence.get('orientation'):
        return 'review', 'missing reviewed visible landmarks and orientation'
    tolerance = float(evidence.get('tolerance_px', 8))
    if not math.isfinite(tolerance) or tolerance <= 0:
        return 'review', 'invalid landmark tolerance'
    mirrored = swap(points, pairs)
    votes, used = [], set()
    for anchor in evidence.get('landmarks', []):
        i = int(anchor['id'])
        pair = next((p for p in pairs if i in p), None)
        if pair is None or pair in used or not anchor.get('visible', False):
            continue
        if points[i][2] <= 0 or mirrored[i][2] <= 0:
            continue
        x, y = float(anchor['x']), float(anchor['y'])
        d0 = math.hypot(points[i][0]-x, points[i][1]-y)
        d1 = math.hypot(mirrored[i][0]-x, mirrored[i][1]-y)
        if d0 <= tolerance and d1 >= 3*tolerance:
            votes.append('correct')
        elif d1 <= tolerance and d0 >= 3*tolerance:
            votes.append('reverse')
        else:
            return 'review', 'landmark does not distinguish orientations'
        used.add(pair)
    if len(votes) < 3 or len(set(votes)) != 1:
        return 'review', 'need three independent symmetry pairs with unanimous landmark evidence'
    return votes[0], 'three or more independent visible landmark pairs agree'


def yolo_rows(raw, count, dims):
    rows = []
    for line in raw.splitlines(keepends=True):
        tokens = re.findall(r'\S+', line)
        if not tokens:
            continue
        if len(tokens) != 5 + count*dims:
            raise ValueError('YOLO row length differs from kpt_shape; refusing to guess')
        chunks = [tokens[5+i*dims:5+(i+1)*dims] for i in range(count)]
        rows.append((line, chunks))
    return rows


def replace_yolo(raw, count, dims, pairs):
    # Replace only token spans; retain bbox, whitespace, newline and numeric lexemes.
    result = []
    for line in raw.splitlines(keepends=True):
        spans = list(re.finditer(r'\S+', line))
        if not spans:
            result.append(line)
            continue
        chunks = [spans[5+i*dims:5+(i+1)*dims] for i in range(count)]
        perm = list(range(count))
        for a, b in pairs:
            perm[a], perm[b] = perm[b], perm[a]
        replacements = []
        for i in range(count):
            for j in range(dims):
                replacements.append((chunks[i][j].start(), chunks[i][j].end(), chunks[perm[i]][j].group()))
        for start, end, value in reversed(replacements):
            line = line[:start] + value + line[end:]
        result.append(line)
    return ''.join(result)


def visualize(image, before, after, path, status):
    from PIL import Image, ImageDraw
    im = Image.open(image).convert('RGB')
    canvas = Image.new('RGB', (im.width*2, im.height+30), 'black')
    draw = ImageDraw.Draw(canvas)
    for panel, points in enumerate((before, after)):
        canvas.paste(im, (panel*im.width, 30))
        draw.text((panel*im.width+8, 8), ('Before' if panel == 0 else 'After / proposed')+' — '+status, fill='white')
        for row in points:
            for i, (x, y, v, *_) in enumerate(row):
                if v > 0:
                    x, y = x+panel*im.width, y+30
                    draw.ellipse((x-3, y-3, x+3, y+3), fill='yellow')
                    draw.text((x+4, y), str(i), fill='red', stroke_width=1, stroke_fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.thumbnail((1920, 1100))
    canvas.save(path)


def run(source, output, evidence_file=None, apply=False):
    import yaml
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Output must be separate from source, outside its directory tree')
    if output.exists():
        raise ValueError('Output already exists; choose a new run directory')
    evidence = json.loads(Path(evidence_file).read_text('utf-8')) if evidence_file else {}
    output.mkdir(parents=True)
    backup = output/'backup'
    if source.is_file():
        backup.mkdir()
        shutil.copy2(source, backup/source.name)
        root = output/'dataset'
        root.mkdir()
        with zipfile.ZipFile(source) as archive:
            for item in archive.infolist():
                target = (root/item.filename).resolve()
                if root.resolve() not in target.parents:
                    raise ValueError('Unsafe archive path')
            archive.extractall(root)
    else:
        shutil.copytree(source, backup)
        root = output/'dataset'
        shutil.copytree(source, root)
    records, jobs = [], []
    yamls = list(root.rglob('data.yaml'))
    if yamls:
        for config in yamls:
            spec = yaml.safe_load(config.read_text('utf-8'))
            count, dims = spec['kpt_shape']
            if count == 26:
                expected = list(range(26))
                for a, b in EXPORT_PAIRS:
                    expected[a], expected[b] = expected[b], expected[a]
                if spec.get('flip_idx') != expected:
                    raise ValueError('26-point export does not match the verified Roboflow symmetry map')
            if dims not in (2, 3):
                raise ValueError('Only standard YOLO Pose dimensions 2 and 3 are supported')
            for label in config.parent.rglob('labels/*.txt'):
                image = next((p for p in (label.parent.parent/'images').glob(label.stem+'.*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')), None)
                key = image.relative_to(root).as_posix() if image else label.relative_to(root).as_posix()
                raw = label.read_bytes().decode('utf-8')
                from PIL import Image
                size = Image.open(image).size if image else (1, 1)
                rows = yolo_rows(raw, count, dims)
                points = [[[float(c[0])*size[0], float(c[1])*size[1], float(c[2]) if dims == 3 else 2] for c in chunks] for _, chunks in rows]
                jobs.append((key, image, points, label, raw, count, dims, None))
    else:
        for label in root.rglob('*.json'):
            doc = json.loads(label.read_text('utf-8'))
            if not isinstance(doc, dict) or 'annotations' not in doc or 'images' not in doc:
                continue
            for image_info in doc['images']:
                anns = [a for a in doc['annotations'] if a['image_id'] == image_info['id'] and 'keypoints' in a]
                if not anns:
                    continue
                points = [[a['keypoints'][i:i+3] for i in range(0, len(a['keypoints']), 3)] for a in anns]
                image = (label.parent/image_info['file_name']).resolve()
                if root.resolve() not in image.parents:
                    raise ValueError('COCO image escapes dataset')
                jobs.append((image.relative_to(root).as_posix(), image if image.exists() else None, points, label, doc, None, None, anns))
    modified_coco = {}
    for key, image, points, label, raw, count, dims, anns in jobs:
        # Evidence IDs are exported array positions, not native Roboflow IDs.
        pairs = pairs_for_export(count) if anns is None else PAIRS
        decisions = [decide(p, evidence.get(key, {}).get(str(i)), pairs) for i, p in enumerate(points)]
        status = 'review' if not decisions or any(d[0] == 'review' for d in decisions) or len({d[0] for d in decisions}) > 1 else decisions[0][0]
        after = [swap(p, pairs) for p in points] if status == 'reverse' else points
        if status == 'reverse' and apply:
            if anns is None:
                label.write_bytes(replace_yolo(raw, count, dims, pairs).encode('utf-8'))
            else:
                for ann in anns:
                    chunks = [ann['keypoints'][i:i+3] for i in range(0, len(ann['keypoints']), 3)]
                    ann['keypoints'] = [v for chunk in swap(chunks, pairs) for v in chunk]
                modified_coco[label] = raw
        viz = 'visualizations/'+str(len(records)).zfill(4)+'.jpg'
        if image:
            visualize(image, points, after, output/viz, status)
        records.append({'image': key, 'status': status, 'changed': status == 'reverse' and apply,
                        'reasons': [d[1] for d in decisions], 'visualization': viz if image else None})
    for label, doc in modified_coco.items():
        label.write_text(json.dumps(doc, ensure_ascii=False, indent=2), 'utf-8')
    manifest = []
    for path in sorted(backup.rglob('*')):
        if path.is_file():
            manifest.append({'file': path.relative_to(backup).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {'mode': 'apply' if apply else 'dry-run', 'native_pairs': PAIRS,
              'export_26_pairs': EXPORT_PAIRS, 'export_26_native_ids': ROBOFLOW_ACTIVE_IDS,
              'counts': {s: sum(r['status'] == s for r in records) for s in ('correct', 'reverse', 'review')},
              'changed': sum(r['changed'] for r in records), 'backup_manifest': manifest, 'images': records}
    (output/'report.json').write_text(json.dumps(report, indent=2), 'utf-8')
    (output/'review.json').write_text(json.dumps([r for r in records if r['status'] == 'review'], indent=2), 'utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('output')
    parser.add_argument('--evidence')
    parser.add_argument('--apply', action='store_true', help='Apply proven reversals to the output copy only')
    args = parser.parse_args()
    report = run(args.source, args.output, args.evidence, args.apply)
    print(json.dumps({'counts': report['counts'], 'changed': report['changed']}))


if __name__ == '__main__':
    main()
