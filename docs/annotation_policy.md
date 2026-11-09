# Annotation policy

## Current taxonomy

Detection classes: `ball`, `goalkeeper`, `player`. Original `player- ball` was explicitly removed at the user's request, in a new immutable working version. The affected image is incomplete until reviewed.

Do not encode ball possession as a player detection class. Roles, teams, identities and event states are separate attributes.

## Bounding boxes

Label all in-scope visible on-court players, goalkeepers and balls. Use visible-extent boxes consistently. Do not hallucinate a full body beyond an image boundary. Mark frames with uncertain extent or missing tiny balls for review. Referees, staff and spectators are outside the three-class supervised detector taxonomy; when using a four-class pretrained football checkpoint, preserve its referee role separately.

Every image must be fully inspected before `complete=true`. Empty labels are verified negatives only after review. Unlabeled objects in partially annotated images must not become training negatives. The UI requires source match/group and split assignment before saving a review.

## Import and corrections

YOLO polygon labels are converted to enclosing detection boxes for this application. Polygon originals remain intact. Invalid geometry, duplicate boxes, unreadable files and class mismatches are reported. A working export may remove exact duplicate boxes, but that does not certify the rest of the image.

Dataset reviews are stored separately with timestamps and history. Box drawing adds proposals; explicit save and completeness confirmation establish a reviewed annotation. Class labels must be checked against image content.

## Splits

Use independent matches whenever possible. Keep adjacent frames, repeated events, replays, crops and augmentations in the same group. The original 240-image export contains only a train partition; it cannot establish validation performance. Near-duplicate checks are conservative and can require manual adjudication. The exporter blocks known unresolved cross-split candidates.

Never label model predictions as human ground truth. Annotation proposals enter a fresh dataset version requiring complete review.
