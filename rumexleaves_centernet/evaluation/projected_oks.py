import numpy as np


def get_closest_point_on_segment(p, a, b):
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    closest = a + t * ab
    return closest, np.linalg.norm(p - closest)


def get_closest_point_on_segment_list(p, a, b):
    d_min = float("inf")
    closest_min = None
    for a_i, b_i in zip(a, b):
        closest, d = get_closest_point_on_segment(p, a_i, b_i)
        if d < d_min:
            d_min = d
            closest_min = closest
    return closest_min


def compute_projected_oks(self, imgId, catId, mode):
    if mode == "kp_all":
        ind = [0, 1, 2, 3, 4, 5, 6, 7]
        project_inds = [1, 2, 4, 5, 6]
    if mode == "kp_stem":
        ind = [0, 1, 2, 3]
        project_inds = [1, 2]
    elif mode == "kp_vein":
        ind = [3, 4, 5, 6, 7]
        project_inds = [4, 5, 6]
    elif mode == "kp_inbetween":
        ind = [1, 2, 4, 5, 6]
        project_inds = [1, 2, 4, 5, 6]
    elif mode == "kp_true":
        ind = [0, 3, 7]
        project_inds = []
    ind = np.array(ind)
    ind = np.transpose(np.array([ind * 3, ind * 3 + 1, ind * 3 + 2])).reshape(-1)
    p = self.params
    # dimention here should be Nxm
    gts = self._gts[imgId, catId]
    dts = self._dts[imgId, catId]
    inds = np.argsort([-d["score"] for d in dts], kind="mergesort")
    dts = [dts[i] for i in inds]
    if len(dts) > p.maxDets[-1]:
        dts = dts[0 : p.maxDets[-1]]
    # if len(gts) == 0 and len(dts) == 0:
    if len(gts) == 0 or len(dts) == 0:
        return []
    ious = np.zeros((len(dts), len(gts)))
    sigmas = [0.05 for i in range(int(ind.shape[0] / 3))]
    sigmas = np.array(sigmas)
    # sigmas = p.kpt_oks_sigmas
    vars = (sigmas * 2) ** 2
    k = len(sigmas)
    # compute oks between each detection and ground truth object
    for i, dt in enumerate(dts):
        d = np.array(dt["keypoints"])
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt["keypoints"])
            g_temp = g.copy()
            for project_ind in project_inds:
                t_j = np.array([g[(project_ind) * 3], g[(project_ind) * 3 + 1]])
                t_j_prev = np.array([g[(project_ind - 1) * 3], g[(project_ind - 1) * 3 + 1]])
                t_j_next = np.array([g[(project_ind + 1) * 3], g[(project_ind + 1) * 3 + 1]])
                if np.all((t_j_prev == t_j_next)):
                    continue
                else:
                    a = [t_j_prev, t_j]
                    b = [t_j, t_j_next]
                    p = np.array([d[project_ind * 3], d[project_ind * 3 + 1]])
                    g_temp[project_ind * 3 : project_ind * 3 + 2] = get_closest_point_on_segment_list(p, a, b)
            d_temp = d[ind]
            g = g_temp[ind]
            xd = d_temp[0::3]
            yd = d_temp[1::3]
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt["bbox"]
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2

            if k1 > 0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((k))
                dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
            e = (dx**2 + dy**2) / vars / (gt["area"] + np.spacing(1)) / 2
            if k1 > 0:
                e = e[vg > 0]
            ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
    return ious
