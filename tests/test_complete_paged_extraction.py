import pytest

from oci.extraction.complete_paged import (
    bounded_recursive_reconcile,
    plan_complete_note_pages,
    validate_absolute_citations,
    validate_complete_note_page_plan,
)


def test_page_cores_cover_every_character_once_with_bounded_overlap():
    text = "0123456789" * 4000
    pages = plan_complete_note_pages(text)
    assert "".join(text[p.core_start:p.core_end] for p in pages) == text
    assert all(p.context_end - p.context_start <= 14000 for p in pages)
    validate_complete_note_page_plan(text, pages)


def test_citations_are_absolute_and_overlap_is_deduplicated():
    text = "abc target xyz"
    citation = {"start": 4, "end": 10, "text": "target"}
    assert len(validate_absolute_citations(text, [citation, citation])) == 1
    with pytest.raises(ValueError, match="exact absolute"):
        validate_absolute_citations(text, [{**citation, "text": "wrong!"}])


def test_recursive_reconciliation_accounts_for_all_leaves():
    result, ledger = bounded_recursive_reconcile(list(range(35)), lambda values: sum(values), fan_in=4)
    assert result == sum(range(35))
    assert ledger["leaf_count"] == 35
    assert ledger["every_child_accounted_for_exactly_once"] is True

