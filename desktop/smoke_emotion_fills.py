"""Rendered RusEmoLex fills: curve clipping, opaque palette, gradients and boundaries."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import emotions


def main():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    chunks = [dict(start=1, end=2, leaders=[0]),
              dict(start=3, end=4, leaders=[3, 4]),
              dict(start=5, end=6, leaders=[]),
              dict(start=7, end=8, leaders=[0, 1, 4])]
    emotions.draw_emotion_chunks(ax, chunks, np.full(8, .2))
    ax.set(xlim=(.5, 8.5), ylim=(-1, 1))
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3] / 255

    def pixel(x, y):
        px, py = ax.transData.transform((x, y))
        return pixels[pixels.shape[0] - 1 - int(py), int(px)]

    for x in (1, 3, 5, 7):
        assert np.allclose(pixel(x, .5), 1), 'Fill leaked above curve'
    assert np.allclose(pixel(1, -.4), to_rgb(emotions.COLORS[0]), atol=.01)
    for x, leaders in ((3, [3, 4]), (7, [0, 1, 4])):
        assert np.allclose(pixel(x, -.98), to_rgb(emotions.COLORS[leaders[0]]), atol=.035)
        assert np.allclose(pixel(x, .18), to_rgb(emotions.COLORS[leaders[-1]]), atol=.035)
    assert len(ax.lines) == len(chunks) + 1, 'Lost chunk boundaries'
    assert ax.collections[2].get_hatch() == '///', 'Lost unknown hatch'
    # A changing curve must also clip each image, including negative tone.
    plt.close(fig)
    fig, ax = plt.subplots()
    emotions.draw_emotion_chunks(ax, [dict(start=1, end=2, leaders=[0, 4])], [-.8, .6])
    ax.set(xlim=(.5, 2.5), ylim=(-1, 1))
    fig.canvas.draw()
    assert ax.images[0].get_clip_path() is not None
    fig.savefig('/tmp/sentilex-emotion-fill-clipping.png')
    plt.close(fig)
    fig, ax = plt.subplots()
    emotions.draw_emotion_chunks(ax, [dict(start=1, end=1, leaders=[0, 4])], [-1.])
    fig.canvas.draw()
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
    emotions.draw_leader_matrix(ax, chunks)
    ax.set_xlim(.5, 8.5)
    fig.canvas.draw()
    assert [t.get_text() for t in ax.get_yticklabels()] == list(emotions.LABELS)
    assert len(ax.patches) == 7  # 1 + 2 + unknown + 3
    pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3] / 255
    for x, categories in ((1, [0]), (3, [3, 4]), (7, [0, 1, 4])):
        for category in categories:
            assert np.allclose(pixel(x, category), to_rgb(emotions.COLORS[category]), atol=.01)
    assert ax.patches[3].get_hatch() == '///'
    fig.savefig('/tmp/sentilex-cluster-matrix.png')
    plt.close(fig)
    print('[smoke-emotion-fills] OK')


if __name__ == '__main__':
    main()
