from scripts.prepare_external_vlm_gt import CLASSES, SEEDS, evidence_indices
from scripts.split_traffic_compilation import CUT_FRAMES


def test_seeds_are_in_source_intervals_and_native_bounds():
    assert len(SEEDS) == len(CLASSES) == 15
    assert sum(len(seeds) for seeds in SEEDS.values()) == 17
    for case_index, seeds in enumerate(SEEDS.values()):
        start, end = CUT_FRAMES[case_index:case_index + 2]
        selected_frames = {start + frame for frame in evidence_indices(end - start)}
        for source_frame, bbox, note in seeds:
            assert source_frame in selected_frames
            left, top, width, height = bbox
            assert left >= 0 and top >= 0 and width > 0 and height > 0
            assert left + width <= 640 and top + height <= 360
            assert note


def test_evidence_sampling_does_not_use_cut_edges():
    for start, end in zip(CUT_FRAMES, CUT_FRAMES[1:]):
        indices = evidence_indices(end - start)
        assert len(indices) == 9
        assert 0 < indices[0] < indices[-1] < end - start - 1
