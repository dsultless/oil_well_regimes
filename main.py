"""Анализ режимов работы нефтяных скважин (тема 27, Final Project).

Запуск:
    python main.py          - интерактивное меню
    python main.py --all    - выполнить все шаги подряд
"""

import json
import sys
from datetime import datetime

from src import config, detectors, pipeline, transform

MENU = """
========= Анализ режимов работы скважин (NK) =========
1 - Подготовить данные (скачать, чанки, оптимизация, Parquet)
2 - Аудит качества данных
3 - Создать базу SQLite
4 - Выполнить SQL-запросы
5 - Разведочный анализ (EDA) и графики
6 - Сравнить методы поиска смены режима
7 - Найти режимы во всех скважинах и сравнить типы
8 - Анализ одной скважины (ввод вручную)
9 - Выполнить все шаги подряд
0 - Выход
"""

ACTIONS = {
    "1": pipeline.step_prepare,
    "2": pipeline.step_quality,
    "3": pipeline.step_database,
    "4": pipeline.step_queries,
    "5": pipeline.step_eda,
    "6": pipeline.step_compare_methods,
    "7": pipeline.step_detect_all,
    "9": pipeline.run_all,
}


def ask_choice(prompt, options):
    """Спрашивает пользователя, пока он не введёт один из вариантов."""
    while True:
        answer = input(prompt).strip()
        if answer in options:
            return answer
        print(f"Нет такого варианта. Доступно: {', '.join(options)}")


def load_best_params():
    """Читает подобранные параметры методов; если их нет, берёт первые из сетки."""
    try:
        with open(config.TABLES / "best_params.json", encoding="utf-8") as f:
            return json.load(f)["best_params"]
    except FileNotFoundError:
        print("Параметры ещё не подобраны (пункт 6), использую значения по умолчанию.")
        return {method: grid[0] for method, grid in detectors.PARAM_GRID.items()}


def analyze_one_well():
    """Пользователь выбирает скважину, параметр и метод, программа показывает режимы
    и сохраняет результат в results/tables/user_analyses.json."""
    df = pipeline.load_clean()
    wells = sorted(df["well_id"].astype(str).unique())
    print("Скважины:", ", ".join(wells))
    well = ask_choice("Введите номер скважины (например NK-7): ", wells)

    well_type = str(df.loc[df["well_id"] == well, "well_type"].iloc[0])
    parameters = config.PARAMS_BY_TYPE[well_type]
    print("Параметры:", ", ".join(parameters))
    parameter = ask_choice("Введите параметр: ", parameters)

    methods = list(detectors.METHODS)
    for number, method in enumerate(methods, start=1):
        print(f"{number} - {detectors.METHOD_TITLES[method]}")
    method = methods[int(ask_choice("Номер метода: ", [str(i) for i in range(1, len(methods) + 1)])) - 1]

    series = transform.make_series(df, well, parameter)
    if len(series) < 20:
        print("Слишком мало данных для анализа.")
        return
    params = load_best_params()[method]
    found = detectors.detect(series.to_numpy(), method, params)
    parts = detectors.segments(series, found)

    print(f"\n{well}, {parameter}, метод: {detectors.METHOD_TITLES[method]}, параметры: {params}")
    print(f"Найдено смен режима: {len(found)}")
    for part in parts:
        print(f"  Режим {part['regime']}: {part['start']:%Y-%m-%d %H:%M} - {part['end']:%Y-%m-%d %H:%M}, "
              f"среднее {part['mean']:.2f}, точек {part['points']}")

    record = {"time": datetime.now().isoformat(timespec="seconds"), "well_id": well,
              "parameter": parameter, "method": method, "params": params,
              "regimes": [{**p, "start": str(p["start"]), "end": str(p["end"])} for p in parts]}
    path = config.TABLES / "user_analyses.json"
    history = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("Файл истории повреждён, начинаю новую историю.")
    history.append(record)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Результат сохранён в {path}")


def main():
    """Главное меню программы."""
    config.ensure_dirs()
    if "--all" in sys.argv:
        pipeline.run_all()
        return
    while True:
        print(MENU)
        choice = input("Выберите пункт: ").strip()
        if choice == "0":
            print("До свидания!")
            break
        elif choice == "8":
            action = analyze_one_well
        elif choice in ACTIONS:
            action = ACTIONS[choice]
        else:
            print("Неверный пункт меню, введите число от 0 до 9.")
            continue
        try:
            action()
        except FileNotFoundError as error:
            print(f"Нет нужного файла: {error}")
        except KeyboardInterrupt:
            print("\nШаг прерван пользователем.")
        except Exception as error:
            print(f"Ошибка при выполнении шага: {error}")


if __name__ == "__main__":
    main()
