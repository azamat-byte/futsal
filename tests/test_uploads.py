import io

from futsal.uploads import save_upload, asset_label, preferred_video


def test_uploaded_video_keeps_original_display_name_and_bytes(tmp_path):
    upload = io.BytesIO(b"video fixture bytes")
    upload.name = "Новый матч France.mp4"
    target = save_upload(upload, tmp_path / "uploads")
    assert target.read_bytes() == upload.getvalue()
    assert asset_label(target) == "Новый матч France.mp4 · uploaded"
    assert save_upload(upload, tmp_path / "uploads") == target
    assert len(list((tmp_path / "uploads").glob("*.mp4"))) == 1


def test_legacy_upload_is_visible_and_preferred_without_reupload(tmp_path):
    original = tmp_path / "existing_match.mp4"
    original.write_bytes(b"original")
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    legacy = uploads / "3294d0ceba49fe1c.mp4"
    legacy.write_bytes(b"saved upload")
    assert asset_label(legacy).startswith("Uploaded video")
    assert preferred_video([original, legacy]) == legacy
    assert preferred_video([original]) == original
