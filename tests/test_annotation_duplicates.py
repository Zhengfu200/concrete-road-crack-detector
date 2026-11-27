from copy import deepcopy

from annotate import unique_review_rows


def record(identifier, **overrides):
    return {"id": identifier, "sha256": "same-image", "source": "source-a",
            "reviewed": False, "excluded": False, "boxes": [], **overrides}


def test_identical_content_is_shown_once_without_changing_annotations():
    rows = [record("a", reviewed=True, boxes=[[0,.5,.5,.4,.4]]),
            record("b", reviewed=True, boxes=[[0,.5,.5,.8,.8]])]
    before = deepcopy(rows)
    assert unique_review_rows(rows) == [rows[0]]
    assert rows == before  # Audit must still be able to see and report both annotations.


def test_reviewed_nonexcluded_record_preferred_over_excluded_or_unreviewed_alias():
    rows = [record("excluded", reviewed=True, excluded=True),
            record("unreviewed"), record("reviewed", reviewed=True)]
    assert unique_review_rows(rows) == [rows[2]]


def test_source_filter_reuses_existing_review_for_identical_image():
    rows = [record("reviewed", reviewed=True), record("alias", source="source-b")]
    assert unique_review_rows(rows, source="source-b") == [rows[0]]
    assert unique_review_rows(rows, source="missing") == []


def test_draft_preferred_and_missing_hash_does_not_merge_different_ids():
    rows = [record("original"), record("draft", annotation_origin="human_review"),
            record("no-hash-a", sha256=None), record("no-hash-b", sha256=None)]
    assert [r["id"] for r in unique_review_rows(rows)] == ["draft", "no-hash-a", "no-hash-b"]
