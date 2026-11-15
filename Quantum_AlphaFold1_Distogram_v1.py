# Quantum_AlphaFold1_Distogram
# adaptacija AlphaFoldove ideje za Loto, sa kvantnim distogramima


#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Quantum AlphaFold1 Distogram v4.


    - 7 apsolutnih kvantnih faktora;
    - svih 21 parnih kvantnih faktora;
    - lokalna simulacija amplituda;
    - cela istorija za zavrsno ucenje;
    - hronoloska validacija i test;
    - iscrpna NEXT pretraga.


    - uci se tacan NLL cele sortirane kombinacije;
    - NEXT koristi istu zajednicku verovatnocu;
    - nema pseudo-NLL-a;
    - nema deljenja referentnim verovatnocama;
    - nema naknadne temperature ili jacine parova.

Raspodela:
    P(y | istorija)
        = proizvod_j P(y_j | y_0,...,y_(j-1), istorija)

Svaka uslovna raspodela ukljucuje:
    - apsolutni Bornov faktor za poziciju j;
    - sve parne Bornove faktore (i,j), i < j;
    - broj mogucih dovrsetaka kombinacije.

Broj dovrsetaka uklanja strukturnu prednost kratkih nastavaka.
Kod ravnih faktora sve kompletne kombinacije imaju istu masu.
To je pocetno stanje; parametri raspodele uce se iz podataka.

Za NEXT se ne bira pohlepno najbolji sledeci broj.
Obilaze se sve kompletne kombinacije preko stabla prefiksa.
Proverava se da zbir njihovih verovatnoca iznosi jedan.

Ogranicenje najvise cetiri uzastopna broja vazi samo za NEXT.
Istorijski podaci se ne filtriraju.

Kvantno se simuliraju lokalna sest-qubitna kola faktora.
Obuka, normalizacija uslovnih raspodela i pretraga su klasicni.
Ne simulira se gust vektor od 2^42 amplituda.
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
from qiskit.circuit.library import DiagonalGate, QFTGate
from qiskit.quantum_info import Statevector


random.seed(39)
np.random.seed(39)


FILES = (
    Path("/data/loto7_4720_k91.csv"),
    Path("/data/loto7_4720_k91_loto_2982.csv"),
    Path("/data/loto7_4720_k91_loto_plus_1738.csv"),
)

N = 39
K = 7
QUBITS = 6
DIM = 1 << QUBITS
SPAN = N - K + 1

PAIRS = tuple(combinations(range(K), 2))
FACTORS = K + len(PAIRS)

FEATURES = 4
HARMONICS = 8
PHASE_TERMS = 2 * HARMONICS
PARAMETER_SHAPE = (FACTORS, FEATURES, PHASE_TERMS)

REGULARIZATIONS = (0.1, 0.01)

TRAIN_FRACTION = 0.70
VALIDATION_END = 0.85

MAX_RUN = 4
TRAIN_BATCH = 256
PREFIX_BATCH = 4096

GRADIENT_TOLERANCE = 2e-5
MAX_OPTIMIZER_ITERATIONS = 2400
DIAGNOSTICS = False

GRID = np.arange(DIM, dtype=np.float64)
BASE_PHASE = np.pi * GRID * GRID / DIM

BASIS = np.asarray(
    [
        function(2.0 * np.pi * harmonic * GRID / DIM)
        for harmonic in range(1, HARMONICS + 1)
        for function in (np.cos, np.sin)
    ],
    dtype=np.float64,
)

PENALTY = np.ones(PARAMETER_SHAPE, dtype=np.float64)
PENALTY[:, 0, :] = 0.1


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def pair_factor(i, j, k=K):
    require(0 <= i < j < k, "Neispravne pozicije para.")
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
    indices = factor_indices(draws)
    scaled = 2.0 * indices / (SPAN - 1) - 1.0

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
    coefficients = np.einsum(
        "bfc,fcp->bfp", features, parameters, optimize=True
    )

    phase = coefficients @ BASIS
    phase += BASE_PHASE[None, None, :]

    diagonal = np.exp(1j * phase)
    amplitudes = np.fft.fft(diagonal, axis=-1) / DIM
    probabilities = np.abs(amplitudes) ** 2

    valid = probabilities[..., :SPAN]
    mass = valid.sum(axis=-1)

    require(
        np.all(np.isfinite(valid))
        and np.all(valid > 0.0)
        and np.all(mass > 0.0),
        "Nulta ili neispravna Bornova verovatnoca.",
    )

    potentials = np.log(valid) - np.log(mass)[..., None]

    return diagonal, amplitudes, probabilities, mass, potentials


@lru_cache(maxsize=None)
def completion_logs(n, k):
    """
    Ako je na poziciji j izabran broj v, preostaje:
        C(n-v, k-j-1)
    mogucih dovrsetaka.

    Ovo je prebrojavanje dozvoljenih nastavaka, ne filter.
    """
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
    """
    Tacna raspodela naredne pozicije, uslovljena samo prefiksom.

    potentials:
        factors x span
        ili batch x factors x span

    prefix:
        batch x position
    """
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
        offsets = (
            candidates[None, :]
            - prefix[:, previous, None]
            - (position - previous)
        )

        # Indeksi nedozvoljenih kandidata koriste se samo kao
        # bezbedni indeksi; ti kandidati zatim dobijaju -inf.
        offsets = np.clip(offsets, 0, span - 1)

        scores += local[
            rows,
            pair_factor(previous, position, k),
            offsets,
        ]

    if position:
        legal = candidates[None, :] > prefix[:, -1, None]
    else:
        legal = np.ones((count, span), dtype=bool)

    require(
        np.all(np.any(legal, axis=1)),
        "Prefiks nema nijedan dozvoljen nastavak.",
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
        log_probabilities = prefix_log_probabilities(
            potentials,
            draws[:, :position],
            position,
            n,
            k,
        )

        indices = draws[:, position] - position - 1
        result += log_probabilities[rows, indices]

    require(
        np.all(np.isfinite(result)),
        "Neispravna verovatnoca kompletne kombinacije.",
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
        log_probabilities = prefix_log_probabilities(
            potentials,
            draws[:, :position],
            position,
            n,
            k,
        )

        observed = draws[:, position] - position - 1
        loss -= float(np.sum(log_probabilities[rows, observed]))

        if not gradient:
            continue

        residual = np.exp(log_probabilities)
        residual[rows, observed] -= 1.0

        derivative[:, position, :] = residual

        for previous in range(position):
            # Parni ishod d=(position-previous)+offset
            # odgovara kandidatu v=y_previous+d.
            candidate_indices = (
                draws[:, previous, None]
                + (position - previous)
                + offsets
                - position
                - 1
            )

            valid = (
                (candidate_indices >= 0)
                & (candidate_indices < span)
            )

            safe_indices = np.clip(
                candidate_indices, 0, span - 1
            )

            derivative[
                :, pair_factor(previous, position, k), :
            ] = (
                residual[gather_rows, safe_indices] * valid
            )

    require(np.isfinite(loss), "Neispravan zajednicki NLL.")

    if gradient:
        require(
            np.all(np.isfinite(derivative)),
            "Neispravan izvod zajednickog NLL-a.",
        )
        return loss, derivative

    return loss


def objective(vector, features, draws, regularization):
    parameters = np.asarray(vector, dtype=np.float64).reshape(
        PARAMETER_SHAPE
    )

    count = len(draws)
    loss = 0.0
    gradient = np.zeros_like(parameters)

    for start in range(0, count, TRAIN_BATCH):
        stop = min(start + TRAIN_BATCH, count)
        x = features[start:stop]
        y = draws[start:stop]

        diagonal, amplitudes, probabilities, mass, potentials = (
            quantum_forward(parameters, x)
        )

        batch_loss, log_derivative = joint_loss(
            potentials, y, gradient=True
        )
        loss += batch_loss

        normalizer_derivative = log_derivative.sum(
            axis=-1, keepdims=True
        )

        amplitude_adjoint = np.zeros_like(amplitudes)
        amplitude_adjoint[..., :SPAN] = amplitudes[..., :SPAN] * (
            log_derivative / probabilities[..., :SPAN]
            - normalizer_derivative / mass[..., None]
        )

        diagonal_adjoint = np.fft.ifft(
            amplitude_adjoint, axis=-1
        )

        phase_derivative = 2.0 * np.imag(
            np.conjugate(diagonal) * diagonal_adjoint
        )
        coefficient_derivative = phase_derivative @ BASIS.T

        gradient += np.einsum(
            "bfc,bfp->fcp",
            x,
            coefficient_derivative,
            optimize=True,
        )

    # Numericko skaliranje cilja po poziciji.
    # Validacija i test prikazuju NLL cele kombinacije.
    denominator = count * K
    loss /= denominator
    gradient /= denominator

    penalty_scale = regularization / FACTORS

    loss += 0.5 * penalty_scale * float(
        np.sum(PENALTY * parameters * parameters)
    )
    gradient += penalty_scale * PENALTY * parameters

    require(
        np.isfinite(loss) and np.all(np.isfinite(gradient)),
        "Nefinitan cilj ili gradijent.",
    )

    return loss, gradient.ravel()


def parameter_coordinates(features, regularization):
    gram = np.einsum(
        "bfc,bfd->fcd", features, features, optimize=True
    ) / len(features)

    metric = gram + regularization * np.diag(
        [0.1, 1.0, 1.0, 1.0]
    )[None, :, :]

    eigenvalues, eigenvectors = np.linalg.eigh(metric)

    require(
        np.all(np.isfinite(eigenvalues))
        and np.all(eigenvalues > 0.0),
        "Metrika skaliranja nije pozitivno definitna.",
    )

    scales = np.sqrt(FACTORS / eigenvalues)
    transform = (
        eigenvectors * scales[:, None, :]
    ) @ np.swapaxes(eigenvectors, -1, -2)

    inverse = np.linalg.inv(transform)

    require(
        np.allclose(
            transform @ inverse,
            np.eye(FEATURES)[None, :, :],
            rtol=1e-10,
            atol=1e-10,
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
        if (
            "eta" in cache
            and np.array_equal(vector, cache["eta"])
        ):
            return cache["value"], cache["gradient"]

        theta = transform @ np.asarray(vector).reshape(
            PARAMETER_SHAPE
        )

        value, raw_gradient = objective(
            theta.ravel(), features, draws, regularization
        )
        raw_gradient = raw_gradient.reshape(PARAMETER_SHAPE)

        transformed_gradient = (
            np.swapaxes(transform, -1, -2) @ raw_gradient
        ).ravel()

        norm = float(np.max(np.abs(raw_gradient)))

        cache.clear()
        cache.update(
            eta=np.asarray(vector).copy(),
            theta=theta.copy(),
            value=float(value),
            gradient=transformed_gradient.copy(),
            raw_norm=norm,
        )

        if (
            not best
            or value < best["value"]
            or (
                value <= best["value"] + 1e-12
                and norm < best["raw_norm"]
            )
        ):
            best.clear()
            best.update(
                theta=theta.copy(),
                value=float(value),
                raw_norm=norm,
            )

        return value, transformed_gradient

    transformed_objective(eta0)

    if cache["raw_norm"] <= GRADIENT_TOLERANCE:
        return cache["theta"].copy()

    def progress(vector):
        nonlocal iterations, last_report
        iterations += 1
        transformed_objective(vector)

        if cache["raw_norm"] <= GRADIENT_TOLERANCE:
            accepted["theta"] = cache["theta"].copy()
            raise StationaryPointFound()

        now = time.monotonic()

        if now - last_report >= 60.0:
            print(
                f"Ucenje: iteracija {iterations}"
                f" | gradijent: {cache['raw_norm']:.6g}",
                flush=True,
            )
            last_report = now

    inverse_transpose = np.swapaxes(inverse, -1, -2)
    gradient_bound = float(
        np.max(np.sum(np.abs(inverse_transpose), axis=-1))
    )

    optimizer_tolerance = (
        0.4 * GRADIENT_TOLERANCE / max(gradient_bound, 1e-12)
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
                "maxiter": MAX_OPTIMIZER_ITERATIONS,
                "gtol": optimizer_tolerance,
                "disp": False,
            },
        )

        transformed_objective(result.x)
        message = str(result.message)

        if cache["raw_norm"] <= GRADIENT_TOLERANCE:
            accepted["theta"] = cache["theta"].copy()

    except StationaryPointFound:
        message = "Dostignut prag izvornog gradijenta."

    theta = (
        accepted["theta"]
        if "theta" in accepted
        else best["theta"]
    )

    _, final_gradient = objective(
        theta.ravel(), features, draws, regularization
    )
    norm = float(np.max(np.abs(final_gradient)))

    require(
        norm <= GRADIENT_TOLERANCE,
        "Optimizacija nije konvergirala;"
        f" gradijent={norm:.6g};"
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
    potentials,
    n=N,
    k=K,
    maximum_run=MAX_RUN,
    announce=True,
):
    """
    Potpuna pretraga stabla prefiksa.

    Ne koristi beam search niti pohlepni izbor.
    Svaka kompletna kombinacija obilazi se tacno jednom.
    """
    total = math.comb(n, k)

    if announce:
        print(f"Pretraga NEXT: svih {total} kombinacija.", flush=True)

    prefixes = np.empty((1, 0), dtype=np.int16)
    prefix_log_mass = np.zeros(1, dtype=np.float64)

    best = None
    best_score = -np.inf
    log_total_mass = -np.inf
    log_allowed_mass = -np.inf

    for position in range(k):
        last = position == k - 1

        expected_children = math.comb(
            n - k + position + 1, position + 1
        )

        if not last:
            next_prefixes = np.empty(
                (expected_children, position + 1),
                dtype=np.int16,
            )
            next_log_mass = np.empty(
                expected_children, dtype=np.float64
            )

        cursor = 0

        for start in range(0, len(prefixes), PREFIX_BATCH):
            stop = min(start + PREFIX_BATCH, len(prefixes))
            block = prefixes[start:stop]

            log_probabilities = prefix_log_probabilities(
                potentials, block, position, n, k
            )

            parent_rows, candidate_indices = np.nonzero(
                np.isfinite(log_probabilities)
            )

            child_count = len(parent_rows)

            children = np.empty(
                (child_count, position + 1),
                dtype=np.int16,
            )

            if position:
                children[:, :position] = block[parent_rows]

            children[:, position] = (
                position + 1 + candidate_indices
            )

            child_scores = (
                prefix_log_mass[start:stop][parent_rows]
                + log_probabilities[parent_rows, candidate_indices]
            )

            require(
                np.all(np.isfinite(child_scores)),
                "Nefinitna verovatnoca u NEXT pretrazi.",
            )

            if not last:
                end = cursor + child_count
                next_prefixes[cursor:end] = children
                next_log_mass[cursor:end] = child_scores
            else:
                log_total_mass = float(
                    np.logaddexp(
                        log_total_mass,
                        logsumexp(child_scores),
                    )
                )

                permitted = allowed_next(children, maximum_run)

                if np.any(permitted):
                    log_allowed_mass = float(
                        np.logaddexp(
                            log_allowed_mass,
                            logsumexp(child_scores[permitted]),
                        )
                    )

                    masked = np.where(
                        permitted, child_scores, -np.inf
                    )
                    index = int(np.argmax(masked))

                    if masked[index] > best_score:
                        best_score = float(masked[index])
                        best = children[index].astype(np.int64)

            cursor += child_count

        require(
            cursor == expected_children,
            "Pretraga nije obuhvatila sve prefikse.",
        )

        if not last:
            prefixes = next_prefixes
            prefix_log_mass = next_log_mass
        else:
            require(cursor == total, "Broj kombinacija nije tacan.")

    require(best is not None, "Nema dozvoljene NEXT kombinacije.")

    # Sada nije potrebna proizvoljna globalna particiona suma.
    # Zbir cele naucene raspodele mora vec biti jedan.
    require(
        np.isfinite(log_total_mass)
        and abs(log_total_mass) <= 5e-9,
        f"Zajednicka raspodela nije normalizovana: log Z={log_total_mass}.",
    )
    require(
        np.isfinite(log_allowed_mass)
        and log_allowed_mass <= log_total_mass + 1e-10,
        "Neispravna masa dozvoljenog NEXT domena.",
    )
    require(
        best_score <= log_allowed_mass + 1e-10,
        "Neispravna verovatnoca NEXT maksimuma.",
    )

    checked = float(
        joint_log_probabilities(potentials, best[None, :], n)[0]
    )

    require(
        math.isclose(
            checked, best_score, rel_tol=1e-11, abs_tol=1e-10
        ),
        "Ponovna provera NEXT verovatnoce nije prosla.",
    )
    require(
        len(set(best.tolist())) == k
        and np.all((best >= 1) & (best <= n)),
        "NEXT nije legalna kombinacija.",
    )
    require(
        bool(allowed_next(best[None, :], maximum_run)[0]),
        "NEXT krsi pravilo uzastopnih brojeva.",
    )

    return best, log_total_mass, log_allowed_mass


def qiskit_factor_state(parameters, feature):
    phase = BASE_PHASE + (feature @ parameters) @ BASIS

    circuit = QuantumCircuit(QUBITS)
    circuit.h(range(QUBITS))
    circuit.append(
        DiagonalGate(np.exp(1j * phase)), range(QUBITS)
    )
    circuit.append(
        QFTGate(QUBITS).inverse(), range(QUBITS)
    )

    return np.asarray(Statevector.from_instruction(circuit).data)


def internal_checks():
    for number, (i, j) in enumerate(PAIRS):
        require(
            pair_factor(i, j) == K + number,
            "Pogresno indeksiranje parnih faktora.",
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
        0.035 * np.sin(np.arange(np.prod(PARAMETER_SHAPE)) * 0.31)
    ).reshape(PARAMETER_SHAPE)

    _, amplitudes, probabilities, _, potentials = quantum_forward(
        parameters, features[:2]
    )

    for factor in (0, 6, 7, 17, 27):
        expected = qiskit_factor_state(
            parameters[factor], features[0, factor]
        )
        require(
            np.allclose(
                expected, amplitudes[0, factor],
                rtol=0.0, atol=2e-11
            ),
            "Qiskit i lokalna simulacija nisu saglasni.",
        )

    require(
        np.allclose(probabilities.sum(axis=-1), 1.0, atol=2e-12),
        "Kvantna stanja nisu normalizovana.",
    )
    require(
        np.allclose(np.exp(potentials).sum(axis=-1), 1.0, atol=2e-12),
        "Bornove raspodele nisu normalizovane.",
    )

    # Analiticki gradijent tacnog zajednickog NLL-a.
    x = features[:2]
    y = history[1:3]
    vector = parameters.ravel()
    epsilon = 1e-6

    original_value, original_gradient = objective(
        vector, x, y, 0.01
    )

    for shift in (0.3, 1.1, 2.4):
        direction = np.sin(
            np.arange(len(vector)) * 0.73 + shift
        )
        direction /= np.linalg.norm(direction)

        numerical = (
            objective(vector + epsilon * direction, x, y, 0.01)[0]
            - objective(vector - epsilon * direction, x, y, 0.01)[0]
        ) / (2.0 * epsilon)

        require(
            math.isclose(
                numerical,
                float(original_gradient @ direction),
                rel_tol=4e-5,
                abs_tol=4e-6,
            ),
            "Analiticki gradijent zajednickog NLL-a nije tacan.",
        )

    # Invertibilno skaliranje mora sacuvati cilj.
    transform, inverse = parameter_coordinates(x, 0.01)
    eta = inverse @ parameters
    restored = transform @ eta

    require(
        np.allclose(restored, parameters, atol=1e-12),
        "Skaliranje menja parametre.",
    )

    restored_value, restored_gradient = objective(
        restored.ravel(), x, y, 0.01
    )

    require(
        np.isclose(restored_value, original_value, atol=1e-11),
        "Skaliranje menja ciljnu funkciju.",
    )
    require(
        np.allclose(
            restored_gradient, original_gradient,
            rtol=1e-9, atol=1e-10
        ),
        "Skaliranje menja izvorni gradijent.",
    )

    eta_gradient = (
        np.swapaxes(transform, -1, -2)
        @ original_gradient.reshape(PARAMETER_SHAPE)
    )

    direction = np.cos(
        np.arange(eta.size) * 0.43
    ).reshape(PARAMETER_SHAPE)
    direction /= np.linalg.norm(direction)

    numerical = (
        objective(
            (transform @ (eta + epsilon * direction)).ravel(),
            x, y, 0.01,
        )[0]
        - objective(
            (transform @ (eta - epsilon * direction)).ravel(),
            x, y, 0.01,
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

    # Potpuna normalizacija i maksimum na malom domenu.
    n, k = 9, 4
    span = n - k + 1
    factors = k + math.comb(k, 2)

    toy = np.asarray(
        list(combinations(range(1, n + 1), k)),
        dtype=np.int64,
    )

    toy_potentials = 0.3 * np.sin(
        np.arange(factors * span).reshape(factors, span) * 0.37
    )

    joint_scores = joint_log_probabilities(toy_potentials, toy, n)

    require(
        abs(float(logsumexp(joint_scores))) < 1e-12,
        "Zbir zajednickih verovatnoca nije jedan.",
    )

    best, log_z, log_allowed = exhaustive_next(
        toy_potentials, n, k, maximum_run=2, announce=False
    )

    permitted = allowed_next(toy, 2)
    expected_index = int(
        np.argmax(np.where(permitted, joint_scores, -np.inf))
    )

    require(
        np.array_equal(best, toy[expected_index]),
        "NEXT maksimum nije tacan.",
    )
    require(abs(log_z) < 1e-12, "Pretraga menja normalizaciju.")
    require(
        np.isclose(
            log_allowed,
            logsumexp(joint_scores[permitted]),
            atol=1e-12,
        ),
        "Masa dozvoljenog domena nije tacna.",
    )

    # Provera izvoda po svim faktorima.
    observed = toy[[3, 41]]
    batched = np.repeat(toy_potentials[None, :, :], 2, axis=0)

    value, derivative = joint_loss(
        batched, observed, n, gradient=True
    )

    require(
        np.isclose(
            value,
            -joint_log_probabilities(batched, observed, n).sum(),
            atol=1e-12,
        ),
        "NLL i zajednicke verovatnoce nisu saglasni.",
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
        "Izvod po parnim i apsolutnim faktorima nije tacan.",
    )

    # Ravni faktori: nijedna kompletna kombinacija nema
    # prednost samo zbog manjeg broja mogucih nastavaka.
    flat = np.zeros((factors, span), dtype=np.float64)
    flat_scores = joint_log_probabilities(flat, toy, n)

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

    flat_full = np.zeros((FACTORS, SPAN), dtype=np.float64)
    edge_scores = joint_log_probabilities(flat_full, examples)

    require(
        np.allclose(
            edge_scores,
            -math.log(math.comb(N, K)),
            rtol=0.0,
            atol=1e-12,
        ),
        "Postoji vestacka prednost krajeva opsega.",
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
        "Pravilo uzastopnih brojeva nije tacno.",
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
    count = len(y)

    train_end = int(TRAIN_FRACTION * count)
    validation_end = int(VALIDATION_END * count)

    require(
        0 < train_end < validation_end < count,
        "Neispravna hronoloska podela.",
    )

    selected = None
    previous_training_parameters = None

    for regularization in REGULARIZATIONS:
        print(
            f"Ucenje zajednicke raspodele: reg={regularization:g}.",
            flush=True,
        )

        try:
            parameters = fit_model(
                x[:train_end],
                y[:train_end],
                regularization,
                previous_training_parameters,
            )

            previous_training_parameters = parameters.copy()

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
        f"Validacija NLL cele kombinacije: {selected['score']:.6f}"
        f" | reg: {regularization:g}",
        flush=True,
    )
    print("Ucenje modela za test.", flush=True)

    test_parameters = fit_model(
        x[:validation_end],
        y[:validation_end],
        regularization,
        selected["parameters"],
    )

    test_score = evaluate(
        test_parameters,
        x[validation_end:],
        y[validation_end:],
    )

    print(f"Test NLL cele kombinacije: {test_score:.6f}", flush=True)
    print("Zavrsno ucenje: svi istorijski prelazi.", flush=True)

    final_parameters = fit_model(
        x, y, regularization, test_parameters
    )

    final_potentials = quantum_forward(
        final_parameters, features[-1:]
    )[-1][0]

    next_combo, log_mass, log_allowed = exhaustive_next(
        final_potentials
    )

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
        "Provera oznaka sedam registara nije prosla.",
    )

    print(f"NEXT: {next_combo.tolist()}", flush=True)

    if DIAGNOSTICS:
        print(f"Istorijski prelazi: {count}")
        print(f"Log ukupne mase: {log_mass:.12g}")
        print(f"Log mase dozvoljenog NEXT domena: {log_allowed:.9f}")

    return {
        "path": path,
        "next": tuple(next_combo.tolist()),
        "digest": hashlib.sha256(draws.tobytes()).hexdigest(),
        "parameters": final_parameters,
        "potentials": final_potentials,
    }


def check_repeated_results(results):
    for first, second in combinations(results, 2):
        if first["next"] != second["next"]:
            continue

        if first["digest"] == second["digest"]:
            reason = "CSV sadrzaj je identican."
        else:
            parameter_difference = float(
                np.linalg.norm(
                    first["parameters"] - second["parameters"]
                )
            )
            potential_difference = float(
                np.linalg.norm(
                    first["potentials"] - second["potentials"]
                )
            )

            reason = (
                "CSV sadrzaji su razliciti;"
                f" razlika parametara={parameter_difference:.6g};"
                f" razlika potencijala={potential_difference:.6g}."
            )

        print(
            f"PROVERA: isti NEXT za {first['path'].name}"
            f" i {second['path'].name}. {reason}",
            flush=True,
        )


def main():
    print("Quantum AlphaFold1 Distogram V2: tacna zajednicka raspodela.")
    print("Sedam apsolutnih i 21 parni kvantni faktor.")
    print("Ucenje i NEXT koriste istu verovatnocu cele kombinacije.")
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
Quantum AlphaFold1 Distogram: tacna zajednicka raspodela.
Sedam apsolutnih i 21 parni kvantni faktor.
Ucenje i NEXT koriste istu verovatnocu cele kombinacije.
Lokalne kvantne amplitude; klasicna obuka i pretraga.
NEXT: najvise cetiri uzastopna broja.
Pokretanje internih provera.
Interne provere: prosle.

CSV: loto7_4720_k91.csv
Ucenje zajednicke raspodele: reg=0.1.
Ucenje: iteracija 185 | gradijent: 0.000126896
Ucenje zajednicke raspodele: reg=0.01.
Ucenje: iteracija 183 | gradijent: 0.000362137
Validacija NLL cele kombinacije: 16.617560 | reg: 0.1
Ucenje modela za test.
Ucenje: iteracija 158 | gradijent: 2.25559e-05
Test NLL cele kombinacije: 16.613745
Zavrsno ucenje: svi istorijski prelazi.
Ucenje: iteracija 141 | gradijent: 5.60299e-05
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [1, x, 3, y, 11, z, 13]

CSV: loto7_4720_k91_loto_2982.csv
Ucenje zajednicke raspodele: reg=0.1.
Ucenje: iteracija 248 | gradijent: 6.36431e-05
Ucenje zajednicke raspodele: reg=0.01.
Ucenje: iteracija 245 | gradijent: 8.84841e-05
Validacija NLL cele kombinacije: 16.668277 | reg: 0.1
Ucenje modela za test.
Ucenje: iteracija 219 | gradijent: 7.74499e-05
Test NLL cele kombinacije: 16.679191
Zavrsno ucenje: svi istorijski prelazi.
Ucenje: iteracija 196 | gradijent: 4.27778e-05
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [1, x, 3, y, 35, z, 37]

CSV: loto7_4720_k91_loto_plus_1738.csv
Ucenje zajednicke raspodele: reg=0.1.
Ucenje zajednicke raspodele: reg=0.01.
Ucenje: iteracija 331 | gradijent: 0.000338908
Validacija NLL cele kombinacije: 16.878013 | reg: 0.1
Ucenje modela za test.
Test NLL cele kombinacije: 16.750732
Zavrsno ucenje: svi istorijski prelazi.
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [1, x, 3, y, 6, z, 37]
"""



"""
AlphaFold 1: naučiti raspodele odnosa između delova, pa iz njih odrediti celu strukturu.

1. Iz sekvence i podataka o srodnim proteinima uči raspodelu mogućih udaljenosti za svaki par aminokiselinskih ostataka.
2. Predviđene raspodele pretvara u potencijal — funkciju koja ocenjuje koliko je predložena struktura saglasna sa naučenim odnosima.
3. Optimizacijom tog potencijala dobija celovitu prostornu strukturu. Uključuje i informacije o uglovima; parovi se zato ne rešavaju izolovano. 

- Iz istorije učim raspodele odnosa između svih 21 para pozicija u sortiranoj sedmorki, uslovljene prethodnim kolima.
- Naučene raspodele pretvaramo u zajednički potencijal za ocenjivanje kompletne kombinacije.
- NEXT dobijam zajedničkom optimizacijom sedam različitih brojeva. Ne biram najbolji odnos za svaki par zasebno, jer takvi izbori mogu biti međusobno nesaglasni.

Same razlike između brojeva ne određuju njihove apsolutne vrednosti — pomerena sedmorka može imati potpuno iste razlike. 
Zato model mora učiti i zavisnost od konkretnih vrednosti brojeva. 
Proizvod parnih raspodela ne smem automatski proglasiti normalizovanom raspodelom cele kombinacije.

Kvantni pravac je učenje tih uslovnih raspodela preko amplituda, uz lokalnu simulaciju. 

učeni odnosi → zajednički potencijal → celovita NEXT kombinacija 


Model sa kvantno učenim raspodelama za svih 21 par pozicija i dodatnim raspodelama koje određuju apsolutne vrednosti brojeva. 
Nezavisno najbolje parne razlike ne moraju dati saglasnu sedmorku. 


Princip AlphaFolda 1: učene raspodele odnosa → referentno korigovan potencijal → zajednički izbor cele kombinacije. 
Kvantni deo računa amplitude distograma; optimizacija i pretraga kombinacija su klasične. 


Uklanjam deljenje geometrijskom referencom, a svih 28 faktora učim zajedno prema uslovnim verovatnoćama iste raspodele. 
Time faktori više neće biti zasebno obučeni pa tek naknadno spojeni. 
Zadržavam svih 21 parnih odnosa i iscrpnu NEXT pretragu.


- skalira koordinate parametara pomoću isključivo trening podataka;
- vodi jedan neprekinut BFGS postupak;
- koristi prethodno naučene parametre pri nastavku;
- gubitak i prag izvornog gradijenta 2e-5.
"""
