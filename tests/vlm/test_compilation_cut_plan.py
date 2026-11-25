from scripts.split_traffic_compilation import CUT_FRAMES, DESCRIPTIONS


def test_fifteen_editorial_sections():
    assert len(CUT_FRAMES) == 16
    assert len(DESCRIPTIONS) == 15
    assert all(start < end for start, end in zip(CUT_FRAMES, CUT_FRAMES[1:]))
    assert 0 < CUT_FRAMES[0] < CUT_FRAMES[-1] < 19062


def test_sections_retain_contiguous_coverage():
    assert sum(end-start for start,end in zip(CUT_FRAMES,CUT_FRAMES[1:])) == CUT_FRAMES[-1]-CUT_FRAMES[0]
