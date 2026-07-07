import numpy as np
from data_pipeline.vio import essential_relative_pose


def test_pure_forward_translation_recovered():
    rng = np.random.default_rng(0)
    K = np.array([[800., 0, 320.], [0, 800., 240.], [0, 0, 1.]])
    P = rng.uniform(-2, 2, size=(200, 3)); P[:, 2] += 6.0
    def proj(Pw):
        p = (K @ Pw.T).T
        return p[:, :2] / p[:, 2:3]
    pts0 = proj(P)
    pts1 = proj(P - np.array([1.0, 0, 0]))   # source +X == points -X
    R, tdir, inl = essential_relative_pose(pts0, pts1, K)
    assert inl > 150
    assert np.allclose(R, np.eye(3), atol=1e-2)
    assert abs(abs(tdir[0]) - 1.0) < 0.05


def test_rotation_recovered():
    rng = np.random.default_rng(2)
    K = np.array([[800., 0, 320.], [0, 800., 240.], [0, 0, 1.]])
    P = rng.uniform(-2, 2, size=(200, 3)); P[:, 2] += 8.0
    th = np.radians(5.0)
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    def proj(Pw):
        p = (K @ Pw.T).T
        return p[:, :2] / p[:, 2:3]
    pts0 = proj(P)
    pts1 = proj((Ry @ P.T).T)                # pure ~5deg yaw, no translation
    R, tdir, inl = essential_relative_pose(pts0, pts1, K)
    ang = np.degrees(np.arccos((np.trace(R) - 1) / 2))
    assert 3.0 < ang < 7.0                   # recovered ~5 deg rotation
