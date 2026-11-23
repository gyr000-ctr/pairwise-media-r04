"""网格边折叠简化（Garland-Heckbert 二次误差度量 QEM）。

约束与策略：
- 折叠代价严格按二次误差度量 v^T (Q1+Q2) v 计算；
- 动态维护六个轴向极值的顶点计数，保证包围盒严格不变（误差 0）；
- 链接条件杜绝重合/非流形面，零面积检查杜绝数值退化三角形；
- 懒删除堆按代价取边，纯函数式局部状态，线程安全、结果确定；
- 非法输入一律抛 ``ValueError``。
"""

import heapq
import math

__all__ = ["Simplify"]

# 对称 4x4 矩阵按 10 个独立分量存储：
# (00, 01, 02, 03, 11, 12, 13, 22, 23, 33)
_QZERO = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate(verts, faces, target):
    """校验并拷贝输入，返回 (positions[list[list[float]]], faces[list[tuple[int]]])。"""
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError("target 必须为正整数")

    if isinstance(verts, (str, bytes)) or not hasattr(verts, "__iter__"):
        raise ValueError("verts 必须是顶点序列")
    if isinstance(faces, (str, bytes)) or not hasattr(faces, "__iter__"):
        raise ValueError("faces 必须是面序列")

    positions = []
    for vi, v in enumerate(verts):
        if isinstance(v, (str, bytes)) or not hasattr(v, "__len__") or len(v) != 3:
            raise ValueError("第 %d 个顶点不是长度为 3 的坐标" % vi)
        p = []
        for c in v:
            if not _is_number(c):
                raise ValueError("第 %d 个顶点含非数值坐标" % vi)
            f = float(c)
            if not math.isfinite(f):
                raise ValueError("第 %d 个顶点含非有限值（NaN/Inf）" % vi)
            p.append(f)
        positions.append(p)

    n = len(positions)
    if n < 3:
        raise ValueError("顶点数至少为 3")

    face_list = []
    for fi, face in enumerate(faces):
        if isinstance(face, (str, bytes)) or not hasattr(face, "__len__") or len(face) != 3:
            raise ValueError("第 %d 个面不是三角形（需要 3 个索引）" % fi)
        idx = []
        for raw in face:
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValueError("第 %d 个面含非整数索引" % fi)
            if raw < 0 or raw >= n:
                raise ValueError("第 %d 个面索引越界" % fi)
            idx.append(raw)
        if idx[0] == idx[1] or idx[0] == idx[2] or idx[1] == idx[2]:
            raise ValueError("第 %d 个面是退化三角形（索引重复）" % fi)
        face_list.append((idx[0], idx[1], idx[2]))

    if not face_list:
        raise ValueError("faces 不能为空")

    return positions, face_list


def _face_quadric(p0, p1, p2):
    """三角形平面的基本二次曲面 K_p（面积归一化法向）。"""
    ux = p1[0] - p0[0]
    uy = p1[1] - p0[1]
    uz = p1[2] - p0[2]
    vx = p2[0] - p0[0]
    vy = p2[1] - p0[1]
    vz = p2[2] - p0[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm == 0.0:
        return _QZERO
    nx /= norm
    ny /= norm
    nz /= norm
    d = -(nx * p0[0] + ny * p0[1] + nz * p0[2])
    return (
        nx * nx, nx * ny, nx * nz, nx * d,
        ny * ny, ny * nz, ny * d,
        nz * nz, nz * d,
        d * d,
    )


def _quadric_cost(q, x, y, z):
    """v^T Q v，v=(x,y,z,1)。"""
    return (
        q[0] * x * x + q[4] * y * y + q[7] * z * z
        + 2.0 * (q[1] * x * y + q[2] * x * z + q[5] * y * z)
        + 2.0 * (q[3] * x + q[6] * y + q[8] * z)
        + q[9]
    )


def _edge_cost(qu, qv, pu, pv):
    """边的 QEM 标量代价（堆键热路径，避免列表/排序/集合开销）。

    与 ``_candidates`` 取同一最小值：3x3 可解时用最优位置的 v^T Q v，
    否则取两端点代价的较小者。
    """
    q00 = qu[0] + qv[0]
    q01 = qu[1] + qv[1]
    q02 = qu[2] + qv[2]
    q03 = qu[3] + qv[3]
    q11 = qu[4] + qv[4]
    q12 = qu[5] + qv[5]
    q13 = qu[6] + qv[6]
    q22 = qu[7] + qv[7]
    q23 = qu[8] + qv[8]
    q33 = qu[9] + qv[9]

    det = (
        q00 * (q11 * q22 - q12 * q12)
        - q01 * (q01 * q22 - q12 * q02)
        + q02 * (q01 * q12 - q11 * q02)
    )
    scale = max(1.0, abs(q00), abs(q01), abs(q02), abs(q11), abs(q12), abs(q22))
    if abs(det) >= 1e-12 * scale * scale * scale:
        b0, b1, b2 = -q03, -q13, -q23
        det_x = (
            b0 * (q11 * q22 - q12 * q12)
            - q01 * (b1 * q22 - q12 * b2)
            + q02 * (b1 * q12 - q11 * b2)
        )
        det_y = (
            q00 * (b1 * q22 - q12 * b2)
            - b0 * (q01 * q22 - q12 * q02)
            + q02 * (q01 * b2 - b1 * q02)
        )
        det_z = (
            q00 * (q11 * b2 - b1 * q12)
            - q01 * (q01 * b2 - b1 * q02)
            + b0 * (q01 * q12 - q11 * q02)
        )
        x = det_x / det
        y = det_y / det
        z = det_z / det
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            return (
                q00 * x * x + q11 * y * y + q22 * z * z
                + 2.0 * (q01 * x * y + q02 * x * z + q12 * y * z)
                + 2.0 * (q03 * x + q13 * y + q23 * z)
                + q33
            )

    ux, uy, uz = pu
    vx, vy, vz = pv
    cu = (
        q00 * ux * ux + q11 * uy * uy + q22 * uz * uz
        + 2.0 * (q01 * ux * uy + q02 * ux * uz + q12 * uy * uz)
        + 2.0 * (q03 * ux + q13 * uy + q23 * uz)
        + q33
    )
    cv = (
        q00 * vx * vx + q11 * vy * vy + q22 * vz * vz
        + 2.0 * (q01 * vx * vy + q02 * vx * vz + q12 * vy * vz)
        + 2.0 * (q03 * vx + q13 * vy + q23 * vz)
        + q33
    )
    return cu if cu < cv else cv


def _candidates(qu, qv, pu, pv):
    """返回按 QEM 代价升序的候选折叠位置 [(cost, (x,y,z)), ...]。"""
    q = [qu[k] + qv[k] for k in range(10)]
    q00, q01, q02, q03, q11, q12, q13, q22, q23, q33 = q

    opts = []
    # Garland-Heckbert 最优位置：求解左上 3x3 子矩阵
    det = (
        q00 * (q11 * q22 - q12 * q12)
        - q01 * (q01 * q22 - q12 * q02)
        + q02 * (q01 * q12 - q11 * q02)
    )
    scale = max(
        1.0,
        abs(q00), abs(q01), abs(q02),
        abs(q11), abs(q12), abs(q22),
    )
    if abs(det) >= 1e-12 * scale * scale * scale:
        b0, b1, b2 = -q03, -q13, -q23
        det_x = (
            b0 * (q11 * q22 - q12 * q12)
            - q01 * (b1 * q22 - q12 * b2)
            + q02 * (b1 * q12 - q11 * b2)
        )
        det_y = (
            q00 * (b1 * q22 - q12 * b2)
            - b0 * (q01 * q22 - q12 * q02)
            + q02 * (q01 * b2 - b1 * q02)
        )
        det_z = (
            q00 * (q11 * b2 - b1 * q12)
            - q01 * (q01 * b2 - b1 * q02)
            + b0 * (q01 * q12 - q11 * q02)
        )
        x = det_x / det
        y = det_y / det
        z = det_z / det
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            opts.append((_quadric_cost(q, x, y, z), (x, y, z)))

    cu = _quadric_cost(q, pu[0], pu[1], pu[2])
    cv = _quadric_cost(q, pv[0], pv[1], pv[2])
    opts.append((cu, (pu[0], pu[1], pu[2])))
    opts.append((cv, (pv[0], pv[1], pv[2])))

    opts.sort(key=lambda item: (item[0], item[1]))
    # 去重（同一坐标可能重复出现）
    unique = []
    seen = set()
    for cost, p in opts:
        if p not in seen:
            seen.add(p)
            unique.append((cost, p))
    return unique


class Simplify:
    """QEM 边折叠简化器；实例无可变状态，可并发复用。"""

    def simplify(self, verts, faces, target):
        positions, face_list = _validate(verts, faces, target)
        n = len(positions)
        m = len(face_list)

        # 目标面数已满足：直接返回等价拷贝（不修改调用方数据）。
        if target >= m:
            return (
                [tuple(p) for p in positions],
                [tuple(t) for t in face_list],
            )

        # ---- 包围盒与极值计数 ----
        mins = [
            min(positions[i][k] for i in range(n))
            for k in range(3)
        ]
        maxs = [
            max(positions[i][k] for i in range(n))
            for k in range(3)
        ]
        diag2 = sum((maxs[k] - mins[k]) ** 2 for k in range(3))
        area_tol2 = (1e-12 * max(diag2, 1e-24)) ** 2

        min_count = [0, 0, 0]
        max_count = [0, 0, 0]
        for p in positions:
            for k in range(3):
                if p[k] == mins[k]:
                    min_count[k] += 1
                if p[k] == maxs[k]:
                    max_count[k] += 1

        # ---- 拓扑结构 ----
        alive = bytearray(b"\x01") * n
        adj = [set() for _ in range(n)]
        vfaces = [set() for _ in range(n)]
        flist = [list(t) for t in face_list]
        fquad = [_QZERO] * m
        face_count = m

        quad = [[0.0] * 10 for _ in range(n)]

        for fid, t in enumerate(flist):
            a, b, c = t
            vfaces[a].add(fid)
            vfaces[b].add(fid)
            vfaces[c].add(fid)
            adj[a].add(b)
            adj[a].add(c)
            adj[b].add(a)
            adj[b].add(c)
            adj[c].add(a)
            adj[c].add(b)

            kq = _face_quadric(positions[a], positions[b], positions[c])
            fquad[fid] = kq
            for v in (a, b, c):
                qv = quad[v]
                for i in range(10):
                    qv[i] += kq[i]

        # ---- 边堆（懒删除）----
        edge_key = {}
        heap = []
        version = 0

        def push_edge(i, j):
            nonlocal version
            if i > j:
                i, j = j, i
            version += 1
            key = version
            edge_key[(i, j)] = key
            ec = _edge_cost(quad[i], quad[j], positions[i], positions[j])
            # 同代价时优先折叠低化合价边，避免平面网格中少数顶点吞并成
            # 高化合价中心（会让重算量爆炸），同时保持结果确定。
            tie = len(adj[i]) + len(adj[j])
            heap.append((ec, tie, i, j, key))

        for i in range(n):
            for j in adj[i]:
                if j > i:
                    push_edge(i, j)
        heapq.heapify(heap)

        # ---- 主折叠循环 ----
        while face_count > target and heap:
            cost, _tie, u, v, ver = heapq.heappop(heap)
            if not alive[u] or not alive[v]:
                continue
            if edge_key.get((u, v)) != ver:
                continue

            pu = positions[u]
            pv = positions[v]

            # 共享面（同时含 u、v）——折叠时删除
            shared = set()
            for fid in vfaces[u]:
                t = flist[fid]
                if t[0] == v or t[1] == v or t[2] == v:
                    shared.add(fid)
            nshared = len(shared)
            if nshared == 0:
                # 理论上不会发生：邻接关系由面建立
                edge_key.pop((u, v), None)
                continue
            # 不允许跌破目标面数：内部边一次删 2 面（闭合流形允许至多超 1），
            # 边界边一次删 1 面，可精确命中 target。越界则跳过该边，后续
            # 边界边（删 1 面）仍可继续精确收缩到 target。
            if face_count - nshared < target:
                continue

            # 链接条件：折叠后不得产生重复三角形。
            # 公共邻居 w 若不属于任何共享面，则 (u,w) 与 (v,w) 两条边各带一个
            # 面，折叠后这两个面会变成同一个三角形 —— 拒绝该折叠。
            shared_third = set()
            for fid in shared:
                t = flist[fid]
                for w in t:
                    if w != u and w != v:
                        shared_third.add(w)
            link_ok = True
            for w in adj[u]:
                if w in adj[v] and w not in shared_third:
                    link_ok = False
                    break
            if not link_ok:
                continue

            cands = _candidates(quad[u], quad[v], pu, pv)
            chosen = None
            for cand_cost, p in cands:
                # 新位置不得越出原始包围盒
                outside = False
                for k in range(3):
                    if p[k] < mins[k] or p[k] > maxs[k]:
                        outside = True
                        break
                if outside:
                    continue

                # 极值计数模拟：六个轴向极值折叠后仍须至少有一个顶点持有
                ext_ok = True
                for k in range(3):
                    hold_min = min_count[k]
                    hold_max = max_count[k]
                    if pu[k] == mins[k]:
                        hold_min -= 1
                    if pv[k] == mins[k]:
                        hold_min -= 1
                    if pu[k] == maxs[k]:
                        hold_max -= 1
                    if pv[k] == maxs[k]:
                        hold_max -= 1
                    if p[k] == mins[k]:
                        hold_min += 1
                    if p[k] == maxs[k]:
                        hold_max += 1
                    if hold_min < 1 or hold_max < 1:
                        ext_ok = False
                        break
                if not ext_ok:
                    continue

                # 几何合法性：折叠后所有受影响面不得退化（零面积）。
                # 链接条件已保证拓扑不产生重合/非流形面，这里兜底数值退化。
                geo_ok = True
                affected = vfaces[u] | vfaces[v]
                for fid in affected:
                    if fid in shared:
                        continue
                    t = flist[fid]
                    # 折叠后的三个顶点：u 移到 p，v 替换为 u
                    coords = []
                    for w in t:
                        if w == v or w == u:
                            coords.append(p)
                        else:
                            coords.append(positions[w])
                    a0, a1, a2 = coords
                    ex = a1[0] - a0[0]
                    ey = a1[1] - a0[1]
                    ez = a1[2] - a0[2]
                    fx = a2[0] - a0[0]
                    fy = a2[1] - a0[1]
                    fz = a2[2] - a0[2]
                    nx = ey * fz - ez * fy
                    ny = ez * fx - ex * fz
                    nz = ex * fy - ey * fx
                    if nx * nx + ny * ny + nz * nz <= area_tol2:
                        geo_ok = False
                        break
                if geo_ok:
                    chosen = p
                    break

            if chosen is None:
                # 当前无合法折叠位置；边保留，等待拓扑变化后的新版本
                continue

            # ---- 提交折叠：v 并入 u，u 移到 chosen ----
            p = chosen
            for k in range(3):
                if pu[k] == mins[k]:
                    min_count[k] -= 1
                if pv[k] == mins[k]:
                    min_count[k] -= 1
                if pu[k] == maxs[k]:
                    max_count[k] -= 1
                if pv[k] == maxs[k]:
                    max_count[k] -= 1
            positions[u] = [p[0], p[1], p[2]]
            for k in range(3):
                if p[k] == mins[k]:
                    min_count[k] += 1
                if p[k] == maxs[k]:
                    max_count[k] += 1

            # 删除共享面
            for fid in shared:
                t = flist[fid]
                for w in t:
                    vfaces[w].discard(fid)
                flist[fid] = None
                fquad[fid] = _QZERO
                face_count -= 1

            # v 的其余面改写为 u
            for fid in tuple(vfaces[v]):
                t = flist[fid]
                for r in range(3):
                    if t[r] == v:
                        t[r] = u
                vfaces[u].add(fid)
            vfaces[v].clear()

            # 邻接合并（先摘掉 u，避免把 u 自身加进 adj[u] 形成自环）
            adj[u].discard(v)
            adj[v].discard(u)
            for w in adj[v]:
                adj[w].discard(v)
                adj[w].add(u)
                adj[u].add(w)
            adj[v].clear()
            alive[v] = 0
            edge_key.pop((u, v), None)

            # 重算 u 的二次曲面（其入射面集合已变化）
            qu = [0.0] * 10
            for fid in vfaces[u]:
                kq = fquad[fid]
                for i in range(10):
                    qu[i] += kq[i]
            quad[u] = qu

            # u 周围的边代价全部失效，重新入堆
            for w in adj[u]:
                push_edge(u, w)

        # ---- 压缩输出，重映射索引 ----
        remap = [-1] * n
        out_verts = []
        for i in range(n):
            if alive[i]:
                remap[i] = len(out_verts)
                out_verts.append(tuple(positions[i]))

        out_faces = []
        for t in flist:
            if t is not None:
                out_faces.append((remap[t[0]], remap[t[1]], remap[t[2]]))

        return out_verts, out_faces
