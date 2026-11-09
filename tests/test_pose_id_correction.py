import copy
import json

from futsal.pose_id_correction import EXPORT_PAIRS, PAIRS, decide, replace_yolo, run, swap


def test_swap_preserves_complete_records_and_is_involution():
    points = [[i*10, i*7, i % 3, .123+i/100] for i in range(32)]
    original = copy.deepcopy(points)
    changed = swap(points)
    assert points == original
    assert swap(changed) == original
    assert sorted(changed) == sorted(original)
    assert changed[4] == original[4]


def test_verified_roboflow_export_map():
    assert EXPORT_PAIRS == [(0, 5), (20, 22), (1, 6), (2, 7), (21, 23),
                            (3, 8), (14, 19), (15, 17), (16, 18), (25, 24), (13, 11)]
    points = [[i*100, i*50, 2] for i in range(26)]
    evidence = {'reviewed': True, 'orientation': 'confirmed', 'landmarks': [
        {'id': a, 'x': points[b][0], 'y': points[b][1], 'visible': True}
        for a, b in EXPORT_PAIRS[:3]]}
    assert decide(points, evidence, EXPORT_PAIRS)[0] == 'reverse'


def test_insufficient_or_conflicting_evidence_abstains():
    points = [[i*100, i*50, 2] for i in range(32)]
    anchors = [{'id': a, 'x': points[b][0], 'y': points[b][1], 'visible': True} for a, b in PAIRS[:3]]
    evidence = {'reviewed': True, 'orientation': 'confirmed broadcast orientation', 'landmarks': anchors}
    assert decide(points, evidence)[0] == 'reverse'
    assert decide(swap(points), evidence)[0] == 'correct'
    assert decide(points[:26], evidence)[0] == 'review'
    assert decide(points, None)[0] == 'review'
    evidence['landmarks'] = anchors[:2]
    assert decide(points, evidence)[0] == 'review'


def test_yolo_retains_lexemes_whitespace_and_non_keypoint_fields():
    text = '0  0.5 0.50 1 1\t' + ' '.join(f'{i}.000 {i}.50 2' for i in range(32)) + '\r\n'
    changed = replace_yolo(text, 32, 3, PAIRS)
    assert changed.startswith('0  0.5 0.50 1 1\t')
    assert changed.endswith('\r\n')
    assert replace_yolo(changed, 32, 3, PAIRS) == text


def test_coco_apply_preserves_source_and_other_annotations(tmp_path):
    from PIL import Image
    source = tmp_path/'source'
    source.mkdir()
    Image.new('RGB', (400, 200)).save(source/'court.png')
    points = [[i*10, i*5, 2] for i in range(32)]
    doc = {'images': [{'id': 1, 'file_name': 'court.png'}],
           'categories': [{'id': 1, 'name': 'court'}],
           'annotations': [{'id': 2, 'image_id': 1, 'category_id': 1,
                            'bbox': [0, 0, 400, 200], 'score': .87,
                            'keypoints': [v for p in points for v in p]},
                           {'id': 3, 'image_id': 1, 'segmentation': [[1, 2, 3, 4]]}]}
    original = json.dumps(doc).encode()
    (source/'annotations.json').write_bytes(original)
    evidence = {'court.png': {'0': {'reviewed': True, 'orientation': 'confirmed', 'tolerance_px': 3,
                'landmarks': [{'id': a, 'x': points[b][0], 'y': points[b][1], 'visible': True}
                              for a, b in PAIRS[:3]]}}}
    evidence_path = tmp_path/'evidence.json'
    evidence_path.write_text(json.dumps(evidence))
    report = run(source, tmp_path/'out', evidence_path, apply=True)
    assert report['changed'] == 1
    assert (source/'annotations.json').read_bytes() == original
    result = json.loads((tmp_path/'out/dataset/annotations.json').read_text())
    assert result['annotations'][0]['keypoints'] == [v for p in swap(points) for v in p]
    result['annotations'][0]['keypoints'] = doc['annotations'][0]['keypoints']
    assert result == doc
