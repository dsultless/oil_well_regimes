"""Методы поиска смены режима (change-point detection).

Каждый метод получает массив чисел x (ряд одного параметра) и возвращает
список индексов, с которых начинается новый режим.

1. threshold_baseline  - базовый вариант: порог от эталонного периода.
2. rolling_window      - скользящая статистика: сравнение двух соседних окон.
3. binary_segmentation - бинарная сегментация с штрафом (как в библиотеке ruptures).
"""

import numpy as np


def robust_std(x):
    """Устойчивая оценка разброса через медианное отклонение (MAD).
    В отличие от обычного std почти не реагирует на редкие выбросы."""
    x = np.asarray(x, dtype=float)
    mad = np.median(np.abs(x - np.median(x)))
    return 1.4826 * mad


def noise_level(x):
    """Уровень шума ряда: MAD от разностей соседних точек.
    Разности убирают влияние смен уровня, поэтому оценка не завышается."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 1e-6
    sigma = robust_std(np.diff(x)) / np.sqrt(2)
    # Нижняя граница, чтобы почти постоянный ряд не давал деления на ноль
    floor = 0.01 * (abs(np.median(x)) + 1e-6)
    return max(sigma, floor, 1e-6)


def threshold_baseline(x, k=4.0, reference=28, run=4):
    """Базовый метод. Берём первые `reference` точек как эталон (медиана и разброс).
    Если `run` точек подряд отклоняются от эталона больше чем на k разбросов,
    считаем, что начался новый режим, и берём новый эталон с этого места."""
    x = np.asarray(x, dtype=float)
    changes = []
    start = 0
    while start + reference < len(x):
        ref = x[start:start + reference]
        center = np.median(ref)
        spread = max(robust_std(ref), noise_level(x))
        outside = np.abs(x[start + reference:] - center) > k * spread
        found = None
        count = 0
        for i, flag in enumerate(outside):
            count = count + 1 if flag else 0
            if count == run:
                found = start + reference + i - run + 1
                break
        if found is None:
            break
        changes.append(found)
        start = found
    return changes


def rolling_window(x, window=20, threshold=4.0):
    """Скользящая статистика. Для каждой точки t сравниваем среднее окна слева
    (window точек до t) и окна справа (window точек после t).
    Оценка = |разница средних| / уровень шума. Пики выше порога = смены режима."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * window + 1:
        return []
    sigma = noise_level(x)
    cumsum = np.concatenate([[0.0], np.cumsum(x)])
    scores = np.zeros(n)
    t = np.arange(window, n - window + 1)
    left = (cumsum[t] - cumsum[t - window]) / window      # среднее окна слева
    right = (cumsum[t + window] - cumsum[t]) / window     # среднее окна справа
    scores[t] = np.abs(right - left) / sigma

    # Берём локальные максимумы выше порога, не ближе window точек друг к другу
    changes = []
    candidates = np.argsort(scores)[::-1]
    for t in candidates:
        if scores[t] < threshold:
            break
        if all(abs(t - c) >= window for c in changes):
            changes.append(int(t))
    return sorted(changes)


def _segment_cost(cumsum, cumsum_sq, start, end):
    """Стоимость отрезка x[start:end] = сумма квадратов отклонений от его среднего."""
    length = end - start
    total = cumsum[end] - cumsum[start]
    total_sq = cumsum_sq[end] - cumsum_sq[start]
    return total_sq - total ** 2 / length


def binary_segmentation(x, penalty=8.0, min_size=8):
    """Бинарная сегментация. Ищем точку, где разрез ряда на две части сильнее всего
    уменьшает сумму квадратов отклонений. Если выигрыш больше штрафа
    (penalty * шум^2 * ln(n)), режем и повторяем для каждой половины."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * min_size:
        return []
    sigma = noise_level(x)
    limit = penalty * sigma ** 2 * np.log(n)
    cumsum = np.concatenate([[0.0], np.cumsum(x)])
    cumsum_sq = np.concatenate([[0.0], np.cumsum(x ** 2)])

    changes = []
    stack = [(0, n)]
    while stack:
        start, end = stack.pop()
        if end - start < 2 * min_size:
            continue
        whole = _segment_cost(cumsum, cumsum_sq, start, end)
        # Все возможные точки разреза сразу (numpy считает их одним действием)
        cuts = np.arange(start + min_size, end - min_size + 1)
        gains = whole - _segment_cost(cumsum, cumsum_sq, start, cuts) \
                      - _segment_cost(cumsum, cumsum_sq, cuts, end)
        best = int(np.argmax(gains))
        best_gain, best_t = float(gains[best]), int(cuts[best])
        if best_gain > limit:
            changes.append(best_t)
            stack.append((start, best_t))
            stack.append((best_t, end))
    return sorted(changes)


# Методы и их настройки по умолчанию (подобраны на train, см. evaluation.py)
METHODS = {
    "threshold_baseline": threshold_baseline,
    "rolling_window": rolling_window,
    "binary_segmentation": binary_segmentation,
}

# Сетка параметров для подбора на обучающей выборке
PARAM_GRID = {
    "threshold_baseline": [{"k": k} for k in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]],
    "rolling_window": [{"window": w, "threshold": h}
                       for w in [12, 20, 28] for h in [2.0, 3.0, 4.0, 6.0, 8.0]],
    "binary_segmentation": [{"penalty": p} for p in [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]],
}

METHOD_TITLES = {
    "threshold_baseline": "Baseline: порог от эталона",
    "rolling_window": "Скользящие окна",
    "binary_segmentation": "Бинарная сегментация",
}


def detect(x, method, params=None):
    """Запускает выбранный метод с параметрами и возвращает индексы смен режима."""
    if method not in METHODS:
        raise ValueError(f"Неизвестный метод: {method}")
    return METHODS[method](x, **(params or {}))


def segments(series, change_indexes):
    """Делит ряд (pandas Series с датами) на режимы по найденным точкам.
    Возвращает список словарей: начало, конец, среднее, медиана, число точек."""
    bounds = [0] + list(change_indexes) + [len(series)]
    result = []
    for i in range(len(bounds) - 1):
        part = series.iloc[bounds[i]:bounds[i + 1]]
        if len(part) == 0:
            continue
        result.append({
            "regime": i + 1,
            "start": part.index[0],
            "end": part.index[-1],
            "points": int(len(part)),
            "mean": float(part.mean()),
            "median": float(part.median()),
            "std": float(part.std()) if len(part) > 1 else 0.0,
        })
    return result
