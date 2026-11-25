import unittest

from scripts import build_lisa_light_candidate as lisa


def frames(*rows):
    return {f'f{frame}.jpg': {'frame': frame, 'rows': [(tag, 10, 10, 30, 60)]} for frame, tag in rows}


class SequenceTests(unittest.TestCase):
    def test_reads_sequence_and_rejects_duplicate_samples(self):
        self.assertEqual(lisa.sequence_of('Annotations/Annotations/dayTrain/dayClip1/frameAnnotationsBOX.csv'), 'dayClip1')
        self.assertEqual(lisa.sequence_of('Annotations/Annotations/daySequence1/frameAnnotationsBOX.csv'), 'daySequence1')
        self.assertIsNone(lisa.sequence_of('Annotations/Annotations/sample-dayClip6/frameAnnotationsBOX.csv'))
        self.assertIsNone(lisa.sequence_of('Annotations/Annotations/daySequence1/frameAnnotationsBULB.csv'))


class ReadTests(unittest.TestCase):
    def test_groups_rows_by_frame(self):
        text = ('Filename;Annotation tag;Upper left corner X;Upper left corner Y;Lower right corner X;Lower right corner Y;'
                'Origin file;Origin frame number;Origin track;Origin track frame number\n'
                'dayTraining/a--00001.jpg;stop;1;2;3;4;x;7;t;0\n'
                'dayTraining/a--00001.jpg;go;5;6;7;8;x;7;t;0\n')
        result = lisa.read_annotations(text)
        self.assertEqual(list(result), ['a--00001.jpg'])
        self.assertEqual(result['a--00001.jpg']['frame'], 7)
        self.assertEqual(len(result['a--00001.jpg']['rows']), 2)


class FrameFilterTests(unittest.TestCase):
    def test_arrow_rejects_whole_frame(self):
        self.assertFalse(lisa.usable_frame([('stop', 1, 1, 2, 2), ('stopLeft', 3, 3, 4, 4)]))
        self.assertTrue(lisa.usable_frame([('stop', 1, 1, 2, 2), ('go', 3, 3, 4, 4)]))
        self.assertFalse(lisa.usable_frame([]))

    def test_thinning_keeps_more_yellow_and_skips_arrows(self):
        rows = [(frame, 'go') for frame in range(0, 64)] + [(frame, 'warning') for frame in range(100, 116)]
        rows.append((200, 'stopLeft'))
        selected = lisa.thin_frames(frames(*rows), default_stride=32, yellow_stride=8)
        self.assertEqual(selected, ['f0.jpg', 'f32.jpg', 'f100.jpg', 'f108.jpg'])


class ConversionTests(unittest.TestCase):
    def test_converts_and_maps_classes(self):
        row = lisa.yolo_rows([('warning', 0, 0, 128, 96)])[0]
        self.assertEqual(row[0], 1)
        self.assertAlmostEqual(row[1], 0.05)
        self.assertAlmostEqual(row[2], 0.05)
        self.assertAlmostEqual(row[3], 0.1)
        self.assertAlmostEqual(row[4], 0.1)

    def test_clips_only_to_image_and_rejects_degenerate(self):
        self.assertAlmostEqual(lisa.yolo_rows([('go', 1270, 0, 1300, 10)])[0][3], 10 / 1280)
        with self.assertRaises(ValueError):
            lisa.yolo_rows([('go', 1290, 0, 1300, 10)])

    def test_size_uses_640_scale(self):
        self.assertEqual(lisa.short_side_bucket((0, 0, 0, 10 / 1280, 100 / 960)), '<8')
        self.assertEqual(lisa.short_side_bucket((0, 0, 0, 64 / 1280, 100 / 960)), '>=32')


if __name__ == '__main__':
    unittest.main()
