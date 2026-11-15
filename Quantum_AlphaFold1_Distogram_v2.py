# Quantum_AlphaFold1_Distogram V2
# adaptacija AlphaFoldove ideje za Loto, sa kvantnim distogramima


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantum AlphaFold1 Distogram.

Kvantni faktori:
    7 apsolutnih raspodela;
    21 raspodela parnih razlika;
    33 ishoda po faktoru;
    32 nezavisne kontrolisane RY rotacije po faktoru.

Nema Furijeovog ogranicenja niti projekcionog readout-a.

Zajednicka raspodela:
    P(y | istorija) = proizvod uslovnih raspodela pozicija.

Ucenje i NEXT koriste istu verovatnocu kompletne kombinacije.
Sve istorijske kombinacije ostaju u obuci.
NEXT dozvoljava najvise cetiri uzastopna broja.

Lokalno se simuliraju sest-qubitni faktori.
Obuka, povezivanje faktora i iscrpna pretraga su klasicni.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import hashlib
import math
import random
import re
import time
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, XGate
from qiskit.quantum_info import Statevector


random.seed(39)
np.random.seed(39)


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4720_k91.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4720_k91_loto_2982.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4720_k91_loto_plus_1738.csv"),
)

N = 39
K = 7
QUBITS = 6
DIM = 1 << QUBITS
SPAN = N - K + 1

PAIRS = tuple(combinations(range(K), 2))
FACTORS = K + len(PAIRS)
FEATURES = 4

REGULARIZATIONS = (10.0, 1.0, 0.1, 0.01)

TRAIN_FRACTION = 0.70
VALIDATION_END = 0.85

MAX_RUN = 4
TRAIN_BATCH = 256
PREFIX_BATCH = 4096

GRADIENT_TOLERANCE = 2e-5
MAX_ITERATIONS = 2400
DIAGNOSTICS = False


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def make_rotation_tree():
    """
    Bazna stanja 0..32 su dozvoljeni ishodi faktora.
    Bitovi se pripremaju redom q0, q1, ..., q5.

    U svakom cvoru sa dve dostupne grane postoji nezavisna
    RY rotacija. Jednograni nastavci su deterministicki.

    Stablo sa 33 lista ima 32 nezavisna grananja.
    """
    leaves = np.arange(SPAN, dtype=np.int64)

    nodes = []
    forced_one = []
    base = []
    signs = []

    for depth in range(QUBITS):
        prefix_mask = (1 << depth) - 1

        for prefix in range(1 << depth):
            active = (leaves & prefix_mask) == prefix

            if not np.any(active):
                continue

            bit_one = ((leaves >> depth) & 1) == 1
            zero = active & ~bit_one
            one = active & bit_one

            count0 = int(np.count_nonzero(zero))
            count1 = int(np.count_nonzero(one))

            if count0 and count1:
                nodes.append((depth, prefix))

                # Pocetni odnos grana odgovara broju listova.
                z0 = (
                    (count1 - count0)
                    / (2.0 * math.sqrt(count0 * count1))
                )
                base.append(z0)

                row = np.zeros(SPAN, dtype=np.float64)
                row[zero] = -1.0
                row[one] = 1.0
                signs.append(row)

            elif count1:
                forced_one.append((depth, prefix))

    require(len(nodes) == SPAN - 1, "Neispravno stablo rotacija.")

    sign = np.asarray(signs, dtype=np.float64)

    return (
        tuple(nodes),
        tuple(forced_one),
        np.asarray(base, dtype=np.float64),
        sign,
        np.abs(sign),
    )


NODES, FORCED_ONE, BASE_Z, SIGN, VISIT = make_rotation_tree()

ROTATIONS = len(NODES)
PARAMETER_SHAPE = (FACTORS, FEATURES, ROTATIONS)


def pair_factor(i, j, k=K):
    require(0 <= i < j < k, "Neispravan par pozicija.")
    return k + i * (2 * k - i - 1) // 2 + j - i - 1


def read_csv(path):
    require(path.is_file(), f"CSV ne postoji: {path}")

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    require(bool(lines), "CSV je prazan.")

    try:
        dialect = csv.Sniffer().sniff(
            "\n".join(lines[:20]), delimiters=",;\t"
        )
        parsed = csv.reader(lines, dialect)
    except csv.Error:
        parsed = (re.split(r"\s+", line) for line in lines)

    rows = []

    for number, fields in enumerate(parsed, start=1):
        require(
            len(fields) == K,
            f"Red {number}: potrebno je sedam kolona bez hedera.",
        )

        try:
            row = [int(field.strip()) for field in fields]
        except ValueError as exc:
            raise ValueError(
                f"Red {number}: ocekuju se celi brojevi."
            ) from exc

        require(
            all(1 <= value <= N for value in row),
            f"Red {number}: broj izvan opsega 1–39.",
        )
        require(
            len(set(row)) == K,
            f"Red {number}: ponovljen broj unutar kombinacije.",
        )

        rows.append(sorted(row))

    draws = np.asarray(rows, dtype=np.int64)
    require(len(draws) >= 80, "Potrebno je najmanje 80 kola.")
    return draws


def factor_indices(draws, n=N):
    k = draws.shape[1]
    span = n - k + 1

    columns = [
        draws[:, position] - position - 1
        for position in range(k)
    ]
    columns.extend(
        draws[:, j] - draws[:, i] - (j - i)
        for i, j in combinations(range(k), 2)
    )

    result = np.column_stack(columns).astype(np.int64)

    require(
        np.all((result >= 0) & (result < span)),
        "Neispravni indeksi faktora.",
    )
    return result


def build_features(draws):
    scaled = 2.0 * factor_indices(draws) / (SPAN - 1) - 1.0

    result = np.empty(
        (len(draws), FACTORS, FEATURES), dtype=np.float64
    )

    average8 = scaled[0].copy()
    average32 = scaled[0].copy()

    for t, values in enumerate(scaled):
        if t:
            average8 += (2.0 / 9.0) * (values - average8)
            average32 += (2.0 / 33.0) * (values - average32)

        result[t, :, 0] = 1.0
        result[t, :, 1] = values
        result[t, :, 2] = average8
        result[t, :, 3] = average32

    return result


def quantum_forward(parameters, features):
    """
    Ugao svake kontrolisane rotacije:
        angle = pi/2 + atan(z)

    Bornove verovatnoce grana:
        p1 = (1 + z/sqrt(1+z*z))/2
        p0 = (1 - z/sqrt(1+z*z))/2

    Log-verovatnoce listova racunaju se tacno preko putanja
    kroz stablo. Amplitude su pozitivni kvadratni koreni
    tih verovatnoca.

    Nema merenja uzorkovanjem niti slucajnog generisanja.
    """
    z = BASE_Z[None, None, :] + np.einsum(
        "bfc,fcr->bfr", features, parameters, optimize=True
    )

    scale = np.hypot(1.0, z)

    log_probabilities = (
        np.arcsinh(z) @ SIGN
        - (math.log(2.0) + np.log(scale)) @ VISIT
    )

    require(
        np.all(np.isfinite(log_probabilities)),
        "Neispravne kvantne log-verovatnoce.",
    )

    return z, scale, log_probabilities


@lru_cache(maxsize=None)
def completion_logs(n, k):
    span = n - k + 1
    result = np.empty((k, span), dtype=np.float64)

    for position in range(k):
        remaining = k - position - 1

        for offset in range(span):
            value = position + 1 + offset
            result[position, offset] = math.log(
                math.comb(n - value, remaining)
            )

    result.setflags(write=False)
    return result


def prefix_log_probabilities(potentials, prefix, position, n=N, k=K):
    count = len(prefix)
    span = n - k + 1

    require(
        prefix.shape == (count, position),
        "Neispravan oblik prefiksa.",
    )

    if potentials.ndim == 2:
        local = np.broadcast_to(
            potentials[None, :, :],
            (count,) + potentials.shape,
        )
    else:
        local = potentials
        require(len(local) == count, "Nesaglasan broj potencijala.")

    candidates = position + 1 + np.arange(span)
    rows = np.arange(count)[:, None]

    scores = (
        local[:, position, :]
        + completion_logs(n, k)[position][None, :]
    )

    for previous in range(position):
        indices = (
            candidates[None, :]
            - prefix[:, previous, None]
            - (position - previous)
        )

        indices = np.clip(indices, 0, span - 1)

        scores += local[
            rows,
            pair_factor(previous, position, k),
            indices,
        ]

    legal = (
        candidates[None, :] > prefix[:, -1, None]
        if position
        else np.ones((count, span), dtype=bool)
    )

    require(
        np.all(np.any(legal, axis=1)),
        "Prefiks nema dozvoljen nastavak.",
    )

    scores = np.where(legal, scores, -np.inf)
    normalizers = logsumexp(scores, axis=1)

    require(
        np.all(np.isfinite(normalizers)),
        "Neispravna normalizacija prefiksa.",
    )

    return scores - normalizers[:, None]


def joint_log_probabilities(potentials, draws, n=N):
    k = draws.shape[1]
    rows = np.arange(len(draws))
    result = np.zeros(len(draws), dtype=np.float64)

    for position in range(k):
        logp = prefix_log_probabilities(
            potentials, draws[:, :position], position, n, k
        )
        indices = draws[:, position] - position - 1
        result += logp[rows, indices]

    require(
        np.all(np.isfinite(result)),
        "Neispravna zajednicka verovatnoca.",
    )
    return result


def joint_loss(potentials, draws, n=N, gradient=False):
    count, k = draws.shape
    span = n - k + 1

    rows = np.arange(count)
    gather_rows = rows[:, None]
    offsets = np.arange(span)[None, :]

    loss = 0.0
    derivative = np.zeros_like(potentials) if gradient else None

    for position in range(k):
        logp = prefix_log_probabilities(
            potentials, draws[:, :position], position, n, k
        )

        observed = draws[:, position] - position - 1
        loss -= float(np.sum(logp[rows, observed]))

        if not gradient:
            continue

        residual = np.exp(logp)
        residual[rows, observed] -= 1.0
        derivative[:, position, :] = residual

        for previous in range(position):
            candidate_indices = (
                draws[:, previous, None]
                + (position - previous)
                + offsets
                - position - 1
            )

            valid = (
                (candidate_indices >= 0)
                & (candidate_indices < span)
            )
            safe = np.clip(candidate_indices, 0, span - 1)

            derivative[:, pair_factor(previous, position, k), :] = (
                residual[gather_rows, safe] * valid
            )

    require(np.isfinite(loss), "Neispravan zajednicki NLL.")

    if gradient:
        require(
            np.all(np.isfinite(derivative)),
            "Neispravan izvod po faktorima.",
        )
        return loss, derivative

    return loss


def objective(vector, features, draws, regularization):
    parameters = np.asarray(vector, dtype=np.float64).reshape(
        PARAMETER_SHAPE
    )

    loss = 0.0
    gradient = np.zeros_like(parameters)

    for start in range(0, len(draws), TRAIN_BATCH):
        stop = min(start + TRAIN_BATCH, len(draws))
        x = features[start:stop]

        z, scale, potentials = quantum_forward(parameters, x)

        batch_loss, log_derivative = joint_loss(
            potentials, draws[start:stop], gradient=True
        )
        loss += batch_loss

        # Analiticki izvod log-Bornovih verovatnoca.
        z_derivative = (
            (log_derivative @ SIGN.T) / scale
            - (log_derivative @ VISIT.T) * z / (scale * scale)
        )

        gradient += np.einsum(
            "bfc,bfr->fcr", x, z_derivative, optimize=True
        )

    denominator = len(draws) * K
    loss /= denominator
    gradient /= denominator

    # Ista penalizacija statickog i istorijskih delova.
    penalty_scale = regularization / FACTORS
    loss += 0.5 * penalty_scale * float(
        np.sum(parameters * parameters)
    )
    gradient += penalty_scale * parameters

    require(
        np.isfinite(loss) and np.all(np.isfinite(gradient)),
        "Nefinitan cilj ili gradijent.",
    )

    return loss, gradient.ravel()


def parameter_coordinates(features, regularization):
    gram = np.einsum(
        "bfc,bfd->fcd", features, features, optimize=True
    ) / len(features)

    metric = (
        gram
        + regularization * np.eye(FEATURES)[None, :, :]
    )

    values, vectors = np.linalg.eigh(metric)

    require(
        np.all(np.isfinite(values)) and np.all(values > 0.0),
        "Neispravna metrika parametara.",
    )

    transform = (
        vectors * np.sqrt(FACTORS / values)[:, None, :]
    ) @ np.swapaxes(vectors, -1, -2)

    inverse = np.linalg.inv(transform)

    require(
        np.allclose(
            transform @ inverse,
            np.eye(FEATURES)[None, :, :],
            atol=1e-10,
            rtol=1e-10,
        ),
        "Promena koordinata nije invertibilna.",
    )

    return transform, inverse


class StationaryPointFound(Exception):
    pass


def fit_model(features, draws, regularization, starting=None):
    transform, inverse = parameter_coordinates(
        features, regularization
    )

    theta0 = (
        np.zeros(PARAMETER_SHAPE, dtype=np.float64)
        if starting is None
        else np.asarray(starting, dtype=np.float64).copy()
    )
    eta0 = (inverse @ theta0).ravel()

    cache = {}
    best = {}
    accepted = {}
    iterations = 0
    last_report = time.monotonic()

    def transformed_objective(vector):
        if "eta" in cache and np.array_equal(vector, cache["eta"]):
            return cache["value"], cache["gradient"]

        theta = transform @ np.asarray(vector).reshape(
            PARAMETER_SHAPE
        )

        value, raw_gradient = objective(
            theta.ravel(), features, draws, regularization
        )
        raw_gradient = raw_gradient.reshape(PARAMETER_SHAPE)

        derivative = (
            np.swapaxes(transform, -1, -2) @ raw_gradient
        ).ravel()
        norm = float(np.max(np.abs(raw_gradient)))

        cache.clear()
        cache.update(
            eta=np.asarray(vector).copy(),
            theta=theta.copy(),
            value=float(value),
            gradient=derivative.copy(),
            norm=norm,
        )

        if (
            not best
            or value < best["value"]
            or (
                value <= best["value"] + 1e-12
                and norm < best["norm"]
            )
        ):
            best.clear()
            best.update(
                theta=theta.copy(),
                value=float(value),
                norm=norm,
            )

        return value, derivative

    transformed_objective(eta0)

    if cache["norm"] <= GRADIENT_TOLERANCE:
        return cache["theta"].copy()

    def progress(vector):
        nonlocal iterations, last_report
        iterations += 1
        transformed_objective(vector)

        if cache["norm"] <= GRADIENT_TOLERANCE:
            accepted["theta"] = cache["theta"].copy()
            raise StationaryPointFound()

        now = time.monotonic()

        if now - last_report >= 60.0:
            print(
                f"Ucenje: iteracija {iterations}"
                f" | gradijent: {cache['norm']:.6g}",
                flush=True,
            )
            last_report = now

    bound = float(
        np.max(
            np.sum(
                np.abs(np.swapaxes(inverse, -1, -2)),
                axis=-1,
            )
        )
    )

    tolerance = (
        0.4 * GRADIENT_TOLERANCE / max(bound, 1e-12)
    )
    message = ""

    try:
        result = minimize(
            transformed_objective,
            eta0,
            method="BFGS",
            jac=True,
            callback=progress,
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": tolerance,
                "disp": False,
            },
        )

        transformed_objective(result.x)
        message = str(result.message)

        if cache["norm"] <= GRADIENT_TOLERANCE:
            accepted["theta"] = cache["theta"].copy()

    except StationaryPointFound:
        message = "Dostignut prag izvornog gradijenta."

    theta = accepted.get("theta", best["theta"])

    _, gradient = objective(
        theta.ravel(), features, draws, regularization
    )
    norm = float(np.max(np.abs(gradient)))

    require(
        norm <= GRADIENT_TOLERANCE,
        f"Optimizacija nije konvergirala; gradijent={norm:.6g};"
        f" iteracije={iterations}; {message}",
    )

    return theta.copy()


def evaluate(parameters, features, draws):
    loss = 0.0

    for start in range(0, len(draws), TRAIN_BATCH):
        stop = min(start + TRAIN_BATCH, len(draws))
        potentials = quantum_forward(
            parameters, features[start:stop]
        )[-1]
        loss += joint_loss(potentials, draws[start:stop])

    return loss / len(draws)


def allowed_next(combos, maximum_run=MAX_RUN):
    differences = np.diff(combos, axis=1)
    allowed = np.ones(len(combos), dtype=bool)

    for start in range(max(0, combos.shape[1] - maximum_run)):
        allowed &= ~np.all(
            differences[:, start:start + maximum_run] == 1,
            axis=1,
        )

    return allowed


def exhaustive_next(
    potentials, n=N, k=K, maximum_run=MAX_RUN, announce=True
):
    total = math.comb(n, k)

    if announce:
        print(f"Pretraga NEXT: svih {total} kombinacija.", flush=True)

    prefixes = np.empty((1, 0), dtype=np.int16)
    prefix_mass = np.zeros(1, dtype=np.float64)

    best = None
    best_score = -np.inf
    log_total = -np.inf
    log_allowed = -np.inf

    for position in range(k):
        last = position == k - 1
        expected = math.comb(
            n - k + position + 1, position + 1
        )

        if not last:
            next_prefixes = np.empty(
                (expected, position + 1), dtype=np.int16
            )
            next_mass = np.empty(expected, dtype=np.float64)

        cursor = 0

        for start in range(0, len(prefixes), PREFIX_BATCH):
            stop = min(start + PREFIX_BATCH, len(prefixes))
            block = prefixes[start:stop]

            logp = prefix_log_probabilities(
                potentials, block, position, n, k
            )
            rows, columns = np.nonzero(np.isfinite(logp))

            children = np.empty(
                (len(rows), position + 1), dtype=np.int16
            )

            if position:
                children[:, :position] = block[rows]

            children[:, position] = position + 1 + columns
            scores = (
                prefix_mass[start:stop][rows] + logp[rows, columns]
            )

            require(
                np.all(np.isfinite(scores)),
                "Nefinitan skor tokom pretrage.",
            )

            if not last:
                end = cursor + len(rows)
                next_prefixes[cursor:end] = children
                next_mass[cursor:end] = scores
            else:
                log_total = float(
                    np.logaddexp(log_total, logsumexp(scores))
                )

                permitted = allowed_next(children, maximum_run)

                if np.any(permitted):
                    log_allowed = float(
                        np.logaddexp(
                            log_allowed,
                            logsumexp(scores[permitted]),
                        )
                    )

                    masked = np.where(permitted, scores, -np.inf)
                    index = int(np.argmax(masked))

                    if masked[index] > best_score:
                        best_score = float(masked[index])
                        best = children[index].astype(np.int64)

            cursor += len(rows)

        require(cursor == expected, "Nepotpuna pretraga prefiksa.")

        if not last:
            prefixes = next_prefixes
            prefix_mass = next_mass
        else:
            require(cursor == total, "Pogresan broj kombinacija.")

    require(best is not None, "Nema dozvoljene NEXT kombinacije.")
    require(
        np.isfinite(log_total) and abs(log_total) <= 5e-9,
        f"Raspodela nije normalizovana: log Z={log_total}.",
    )
    require(
        np.isfinite(log_allowed)
        and log_allowed <= log_total + 1e-10
        and best_score <= log_allowed + 1e-10,
        "Neispravna masa dozvoljenog domena.",
    )

    checked = float(
        joint_log_probabilities(potentials, best[None, :], n)[0]
    )

    require(
        math.isclose(
            checked, best_score, rel_tol=1e-11, abs_tol=1e-10
        ),
        "Provera NEXT maksimuma nije prosla.",
    )
    require(
        len(set(best.tolist())) == k
        and np.all((best >= 1) & (best <= n))
        and allowed_next(best[None, :], maximum_run)[0],
        "Nelegalna NEXT kombinacija.",
    )

    return best, log_total, log_allowed


def qiskit_factor_state(parameters, feature):
    z = BASE_Z + feature @ parameters
    angles = np.pi / 2.0 + np.arctan(z)

    circuit = QuantumCircuit(QUBITS)

    operations = [
        (depth, prefix, float(angle))
        for (depth, prefix), angle in zip(NODES, angles)
    ]
    operations.extend(
        (depth, prefix, None)
        for depth, prefix in FORCED_ONE
    )
    operations.sort(key=lambda item: (item[0], item[1]))

    for depth, prefix, angle in operations:
        gate = XGate() if angle is None else RYGate(angle)

        if depth == 0:
            circuit.append(gate, [0])
        else:
            controlled = gate.control(
                num_ctrl_qubits=depth,
                ctrl_state=prefix,
            )
            circuit.append(
                controlled,
                list(range(depth)) + [depth],
            )

    return np.asarray(Statevector.from_instruction(circuit).data)


def internal_checks():
    for index, (i, j) in enumerate(PAIRS):
        require(
            pair_factor(i, j) == K + index,
            "Neispravno indeksiranje parova.",
        )

    history = np.asarray(
        [
            [1, 3, 8, 12, 19, 27, 38],
            [2, 6, 9, 15, 21, 30, 39],
            [4, 7, 11, 16, 23, 29, 35],
            [1, 5, 13, 18, 25, 32, 37],
        ],
        dtype=np.int64,
    )

    features = build_features(history)
    parameters = (
        0.07 * np.sin(np.arange(np.prod(PARAMETER_SHAPE)) * 0.31)
    ).reshape(PARAMETER_SHAPE)

    _, _, logp = quantum_forward(parameters, features[:2])

    require(
        np.allclose(logsumexp(logp, axis=-1), 0.0, atol=2e-12),
        "Kvantne raspodele nisu normalizovane.",
    )

    for factor in (0, 6, 7, 17, 27):
        state = qiskit_factor_state(
            parameters[factor], features[0, factor]
        )

        expected = np.zeros(DIM, dtype=np.complex128)
        expected[:SPAN] = np.exp(0.5 * logp[0, factor])

        require(
            np.allclose(state, expected, rtol=0.0, atol=2e-11),
            "Qiskit i lokalne amplitude nisu saglasni.",
        )

    # Potpuna lokalna sloboda 33-ishodne raspodele:
    # Jakobijan po 32 rotacije mora imati rang 32.
    scale = np.hypot(1.0, BASE_Z)
    jacobian = (
        SIGN / scale[:, None]
        - VISIT * (BASE_Z / (scale * scale))[:, None]
    )

    require(
        np.linalg.matrix_rank(jacobian, tol=1e-10) == SPAN - 1,
        "Kvantni faktor nema 32 nezavisna pravca.",
    )

    x = features[:2]
    y = history[1:3]
    vector = parameters.ravel()
    epsilon = 1e-6

    value, gradient = objective(vector, x, y, 0.1)

    for shift in (0.3, 1.1, 2.4):
        direction = np.sin(
            np.arange(len(vector)) * 0.73 + shift
        )
        direction /= np.linalg.norm(direction)

        numerical = (
            objective(vector + epsilon * direction, x, y, 0.1)[0]
            - objective(vector - epsilon * direction, x, y, 0.1)[0]
        ) / (2.0 * epsilon)

        require(
            math.isclose(
                numerical,
                float(gradient @ direction),
                rel_tol=4e-5,
                abs_tol=4e-6,
            ),
            "Analiticki gradijent nije tacan.",
        )

    transform, inverse = parameter_coordinates(x, 0.1)
    eta = inverse @ parameters
    restored = transform @ eta

    restored_value, restored_gradient = objective(
        restored.ravel(), x, y, 0.1
    )

    require(
        np.allclose(restored, parameters, atol=1e-12)
        and np.isclose(restored_value, value, atol=1e-11)
        and np.allclose(
            restored_gradient, gradient, rtol=1e-9, atol=1e-10
        ),
        "Skaliranje menja model ili ciljnu funkciju.",
    )

    eta_gradient = (
        np.swapaxes(transform, -1, -2)
        @ gradient.reshape(PARAMETER_SHAPE)
    )
    direction = np.cos(
        np.arange(eta.size) * 0.43
    ).reshape(PARAMETER_SHAPE)
    direction /= np.linalg.norm(direction)

    numerical = (
        objective(
            (transform @ (eta + epsilon * direction)).ravel(),
            x, y, 0.1,
        )[0]
        - objective(
            (transform @ (eta - epsilon * direction)).ravel(),
            x, y, 0.1,
        )[0]
    ) / (2.0 * epsilon)

    require(
        math.isclose(
            numerical,
            float(np.sum(eta_gradient * direction)),
            rel_tol=5e-5,
            abs_tol=5e-6,
        ),
        "Gradijent u skaliranim koordinatama nije tacan.",
    )

    n, k = 9, 4
    span = n - k + 1
    factors = k + math.comb(k, 2)

    toy = np.asarray(
        list(combinations(range(1, n + 1), k)),
        dtype=np.int64,
    )
    potentials = 0.3 * np.sin(
        np.arange(factors * span).reshape(factors, span) * 0.37
    )

    scores = joint_log_probabilities(potentials, toy, n)

    require(
        abs(float(logsumexp(scores))) < 1e-12,
        "Zajednicka raspodela nije normalizovana.",
    )

    best, log_total, log_allowed = exhaustive_next(
        potentials, n, k, maximum_run=2, announce=False
    )
    permitted = allowed_next(toy, 2)
    expected = int(
        np.argmax(np.where(permitted, scores, -np.inf))
    )

    require(
        np.array_equal(best, toy[expected])
        and abs(log_total) < 1e-12
        and np.isclose(
            log_allowed, logsumexp(scores[permitted]), atol=1e-12
        ),
        "Provera iscrpne pretrage nije prosla.",
    )

    observed = toy[[3, 41]]
    batched = np.repeat(potentials[None, :, :], 2, axis=0)

    loss, derivative = joint_loss(
        batched, observed, n, gradient=True
    )

    require(
        np.isclose(
            loss,
            -joint_log_probabilities(batched, observed, n).sum(),
            atol=1e-12,
        ),
        "NLL nije saglasan sa zajednickom raspodelom.",
    )

    direction = np.cos(
        np.arange(batched.size) * 0.29
    ).reshape(batched.shape)
    direction /= np.linalg.norm(direction)

    numerical = (
        joint_loss(batched + epsilon * direction, observed, n)
        - joint_loss(batched - epsilon * direction, observed, n)
    ) / (2.0 * epsilon)

    require(
        math.isclose(
            numerical,
            float(np.sum(derivative * direction)),
            rel_tol=3e-5,
            abs_tol=3e-6,
        ),
        "Izvod po parnim faktorima nije tacan.",
    )

    flat_scores = joint_log_probabilities(
        np.zeros((factors, span)), toy, n
    )
    require(
        np.allclose(
            flat_scores,
            -math.log(math.comb(n, k)),
            rtol=0.0,
            atol=1e-12,
        ),
        "Pocetna raspodela favorizuje pojedine kombinacije.",
    )

    examples = np.asarray(
        [
            [1, 2, 3, 5, 6, 7, 8],
            [4, 9, 14, 20, 25, 31, 37],
            [32, 33, 34, 36, 37, 38, 39],
        ],
        dtype=np.int64,
    )

    zero_parameters = np.zeros(PARAMETER_SHAPE)
    initial_potentials = quantum_forward(
        zero_parameters, features[:1]
    )[-1][0]

    require(
        np.allclose(
            joint_log_probabilities(initial_potentials, examples),
            -math.log(math.comb(N, K)),
            rtol=0.0,
            atol=2e-11,
        ),
        "Pocetno kvantno stanje favorizuje krajeve opsega.",
    )

    runs = np.asarray(
        [
            [1, 2, 3, 4, 8, 12, 20],
            [1, 2, 3, 4, 5, 12, 20],
            [1, 8, 9, 10, 11, 12, 20],
            [1, 8, 15, 16, 17, 18, 19],
        ],
        dtype=np.int64,
    )

    require(
        np.array_equal(
            allowed_next(runs),
            np.asarray([True, False, False, False]),
        ),
        "Neispravno pravilo uzastopnih brojeva.",
    )
    require(
        np.array_equal(build_features(history[:3]), features[:3]),
        "Karakteristike koriste buduce redove.",
    )


def process_csv(path):
    print()
    print(f"CSV: {path.name}", flush=True)

    draws = read_csv(path)
    features = build_features(draws)

    x = features[:-1]
    y = draws[1:]

    train_end = int(TRAIN_FRACTION * len(y))
    validation_end = int(VALIDATION_END * len(y))

    require(
        0 < train_end < validation_end < len(y),
        "Neispravna hronoloska podela.",
    )

    selected = None
    previous = None

    for regularization in REGULARIZATIONS:
        print(
            f"Ucenje zajednicke raspodele: reg={regularization:g}.",
            flush=True,
        )

        try:
            parameters = fit_model(
                x[:train_end], y[:train_end],
                regularization, previous,
            )
            previous = parameters.copy()

            score = evaluate(
                parameters,
                x[train_end:validation_end],
                y[train_end:validation_end],
            )

            if selected is None or score < selected["score"]:
                selected = {
                    "score": score,
                    "reg": regularization,
                    "parameters": parameters.copy(),
                }

        except (RuntimeError, FloatingPointError) as exc:
            print(
                f"PROVERA: kandidat reg={regularization:g}"
                f" odbacen: {exc}",
                flush=True,
            )

    require(selected is not None, "Nijedan kandidat nije prosao provere.")

    regularization = selected["reg"]

    print(
        f"Validacija NLL: {selected['score']:.6f}"
        f" | reg: {regularization:g}",
        flush=True,
    )
    print("Ucenje modela za test.", flush=True)

    test_parameters = fit_model(
        x[:validation_end], y[:validation_end],
        regularization, selected["parameters"],
    )
    test_score = evaluate(
        test_parameters,
        x[validation_end:],
        y[validation_end:],
    )

    print(f"Test NLL: {test_score:.6f}", flush=True)
    print("Zavrsno ucenje: svi istorijski prelazi.", flush=True)

    final_parameters = fit_model(
        x, y, regularization, test_parameters
    )
    potentials = quantum_forward(
        final_parameters, features[-1:]
    )[-1][0]

    next_combo, log_total, log_allowed = exhaustive_next(potentials)

    basis_label = sum(
        (int(value) - 1) << (QUBITS * position)
        for position, value in enumerate(next_combo)
    )
    recovered = [
        ((basis_label >> (QUBITS * position)) & (DIM - 1)) + 1
        for position in range(K)
    ]

    require(
        recovered == next_combo.tolist(),
        "Neispravne oznake sedam registara.",
    )

    print(f"NEXT: {next_combo.tolist()}", flush=True)

    if DIAGNOSTICS:
        print(f"Log ukupne mase: {log_total:.12g}")
        print(f"Log dozvoljene mase: {log_allowed:.9f}")

    return {
        "path": path,
        "next": tuple(next_combo.tolist()),
        "digest": hashlib.sha256(draws.tobytes()).hexdigest(),
        "parameters": final_parameters,
        "potentials": potentials,
    }


def check_repeated_results(results):
    for first, second in combinations(results, 2):
        if first["next"] != second["next"]:
            continue

        if first["digest"] == second["digest"]:
            reason = "CSV sadrzaj je identican."
        else:
            dp = float(np.linalg.norm(
                first["parameters"] - second["parameters"]
            ))
            ds = float(np.linalg.norm(
                first["potentials"] - second["potentials"]
            ))
            reason = (
                "CSV sadrzaji su razliciti;"
                f" razlika parametara={dp:.6g};"
                f" razlika potencijala={ds:.6g}."
            )

        print(
            f"PROVERA: isti NEXT za {first['path'].name}"
            f" i {second['path'].name}. {reason}",
            flush=True,
        )


def main():
    print("Quantum AlphaFold1 Distogram V2: nezavisne kvantne rotacije.")
    print("Sedam apsolutnih i 21 parni faktor; tacan zajednicki NLL.")
    print("Lokalne kvantne amplitude; klasicna obuka i pretraga.")
    print("NEXT: najvise cetiri uzastopna broja.")
    print("Pokretanje internih provera.", flush=True)

    try:
        internal_checks()
    except Exception as exc:
        print(f"STOP: interna provera nije prosla: {exc}", flush=True)
        return 1

    print("Interne provere: prosle.", flush=True)

    results = []
    failures = 0

    for path in FILES:
        try:
            results.append(process_csv(path))
        except KeyboardInterrupt:
            print()
            print("Obrada prekinuta korisnickim zahtevom.", flush=True)
            return 130
        except Exception as exc:
            failures += 1
            print(f"STOP za ovaj CSV: {exc}", flush=True)

    check_repeated_results(results)

    if failures:
        print("Obrada nije kompletna: neki CSV-ovi nisu prosli provere.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Quantum AlphaFold1 Distogram V2: nezavisne kvantne rotacije.
Sedam apsolutnih i 21 parni faktor; tacan zajednicki NLL.
Lokalne kvantne amplitude; klasicna obuka i pretraga.
NEXT: najvise cetiri uzastopna broja.
Pokretanje internih provera.
Interne provere: prosle.

CSV: loto7_4720_k91.csv
Ucenje zajednicke raspodele: reg=10.
Ucenje zajednicke raspodele: reg=1.
Ucenje zajednicke raspodele: reg=0.1.
Ucenje zajednicke raspodele: reg=0.01.
Ucenje: iteracija 65 | gradijent: 3.50424e-05
Validacija NLL: 16.549065 | reg: 10
Ucenje modela za test.
Test NLL: 16.549902
Zavrsno ucenje: svi istorijski prelazi.
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [10, 11, 15, 24, 27, 33, 34]

CSV: loto7_4720_k91_loto_2982.csv
Ucenje zajednicke raspodele: reg=10.
Ucenje zajednicke raspodele: reg=1.
Ucenje zajednicke raspodele: reg=0.1.
Ucenje zajednicke raspodele: reg=0.01.
Ucenje: iteracija 69 | gradijent: 2.81255e-05
Validacija NLL: 16.553047 | reg: 10
Ucenje modela za test.
Test NLL: 16.549387
Zavrsno ucenje: svi istorijski prelazi.
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [8, 15, 16, 22, 23, 28, 37]

CSV: loto7_4720_k91_loto_plus_1738.csv
Ucenje zajednicke raspodele: reg=10.
Ucenje zajednicke raspodele: reg=1.
Ucenje zajednicke raspodele: reg=0.1.
Ucenje zajednicke raspodele: reg=0.01.
Ucenje: iteracija 72 | gradijent: 3.53667e-05
Validacija NLL: 16.569669 | reg: 10
Ucenje modela za test.
Test NLL: 16.544460
Zavrsno ucenje: svi istorijski prelazi.
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [7, 8, 11, 23, 27, 34, 37]
"""



"""
Svaki kvantni faktor ima 32 nezavisne rotacije za svojih 33 ishoda, 
bez spektralnog povezivanja susednih brojeva. 
"""
