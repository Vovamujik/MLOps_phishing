"""EDA датасета PhiUSIIL: графики, таблицы и выводы в один ран MLflow.

Запуск: uv run python -m training.eda
"""

import mlflow
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score

from training.data import TARGET
from training.download_data import CSV_PATH, CSV_SHA256, DATASET_URL, sha256_of
from training.tracking import setup_mlflow

EXPERIMENT = "phiusiil-eda"
CLASSES = {0: "legit", 1: "phishing"}
COLORS = {0: "tab:blue", 1: "tab:red"}
TOP_TLD = 15
# С этой колонки начинаются признаки, посчитанные по скачанной HTML-странице
FIRST_PAGE_FEATURE = "LineOfCode"


def load_raw() -> pd.DataFrame:
    """Сырой CSV как есть: с FILENAME и дублями. Цель переводим в is_phishing, как в data.py."""
    if not CSV_PATH.exists() or sha256_of(CSV_PATH) != CSV_SHA256:
        raise SystemExit("Нет CSV или он изменён: uv run python -m training.download_data")
    df = pd.read_csv(CSV_PATH)
    df[TARGET] = 1 - df["label"]
    return df


def url_tail(urls: pd.Series) -> pd.Series:
    """Категория хвоста после хоста: нет «/», только «/» в конце или настоящий путь/query."""
    rest = urls.str.replace(r"^[a-zA-Z]+://", "", regex=True)
    return pd.Series(
        np.select(
            [~rest.str.contains("/", regex=False), rest.str.fullmatch(r"[^/]+/")],
            ["нет /", "только / в конце"],
            default="путь или query",
        ),
        index=urls.index,
    )


def url_prefix(urls: pd.Series) -> pd.Series:
    return urls.str.extract(r"^(https?://(?:www\.)?)", expand=False).fillna("другое")


def filename_says_phishing(filenames: pd.Series) -> pd.Series:
    """У фишинга имя файла с буквенным префиксом (mw, oph, ...) или номер от миллиона."""
    number = pd.to_numeric(filenames.str.extract(r"^(\d+)\.", expand=False), errors="coerce")
    return filenames.str.match(r"^[A-Za-z]") | (number >= 1_000_000)


def feature_auc(df: pd.DataFrame) -> pd.DataFrame:
    """ROC-AUC каждого числового признака в одиночку, без модели.

    auc_any_direction = max(AUC, 1 - AUC): признак полезен, даже если связь обратная.
    """
    numeric = df.select_dtypes("number").drop(columns=["label", TARGET])
    page_features = set(df.columns[df.columns.get_loc(FIRST_PAGE_FEATURE) :])
    y = df[TARGET]
    rows = []
    for column in numeric:
        auc = roc_auc_score(y, numeric[column])
        rows.append(
            {
                "feature": column,
                "source": "page" if column in page_features else "url",
                "roc_auc": round(auc, 4),
                "auc_any_direction": round(max(auc, 1 - auc), 4),
                "median_legit": numeric.loc[y == 0, column].median(),
                "median_phishing": numeric.loc[y == 1, column].median(),
            }
        )
    return pd.DataFrame(rows).sort_values("auc_any_direction", ascending=False, ignore_index=True)


def compute_facts(df: pd.DataFrame) -> dict[str, float]:
    y = df[TARGET]
    phishing = y == 1
    sim_rule = df["URLSimilarityIndex"] != 100
    www = df["URL"].str.startswith("https://www.")
    has_slash = url_tail(df["URL"]) != "нет /"
    return {
        "rows": len(df),
        "legit_rows": int((~phishing).sum()),
        "phishing_rows": int(phishing.sum()),
        "phishing_share": phishing.mean(),
        "constant_model_accuracy": max(phishing.mean(), 1 - phishing.mean()),
        "sim100_legit_share": (~sim_rule[~phishing]).mean(),
        "sim100_phishing_share": (~sim_rule[phishing]).mean(),
        "sim_rule_accuracy": (sim_rule == phishing).mean(),
        "https_www_legit_share": www[~phishing].mean(),
        "https_www_phishing_share": www[phishing].mean(),
        "https_www_rule_accuracy": (~www == phishing).mean(),
        "slash_legit_share": has_slash[~phishing].mean(),
        "slash_phishing_share": has_slash[phishing].mean(),
        "uppercase_urls": int(df["URL"].str.contains("[A-Z]").sum()),
        "filename_rule_accuracy": (filename_says_phishing(df["FILENAME"]) == phishing).mean(),
        "duplicate_urls": int(df["URL"].duplicated().sum()),
    }


def conclusions(facts: dict[str, float], auc: pd.DataFrame) -> str:
    top = auc.head(4)
    top_text = ", ".join(f"{r.feature} {r.auc_any_direction:.3f}" for r in top.itertuples())
    page_in_top10 = (auc.head(10)["source"] == "page").sum()
    return "\n".join(
        [
            "Выводы EDA по PhiUSIIL",
            "",
            f"1. Классы почти сбалансированы: {facts['legit_rows']:,} легитимных и "
            f"{facts['phishing_rows']:,} фишинга ({facts['phishing_share']:.1%}); константная "
            f"модель даёт accuracy {facts['constant_model_accuracy']:.3f}.",
            f"2. Одно правило «URLSimilarityIndex != 100 → фишинг» даёт accuracy "
            f"{facts['sim_rule_accuracy']:.2%}, имя файла (FILENAME) — "
            f"{facts['filename_rule_accuracy']:.2%}: метка зашита в служебные колонки.",
            f"3. Все легитимные URL — главные страницы вида https://www.<домен> "
            f"({facts['https_www_legit_share']:.0%} с https://www., «/» после хоста у "
            f"{facts['slash_legit_share']:.0%}), у фишинга https://www. лишь у "
            f"{facts['https_www_phishing_share']:.1%}, «/» после хоста у "
            f"{facts['slash_phishing_share']:.0%}. Модель учит формат сбора, а не фишинг.",
            f"4. В одиночку почти идеально разделяют классы: {top_text}; из топ-10 признаков "
            f"{page_in_top10} посчитаны по HTML и в основном говорят, что фишинговую страницу "
            "не удалось скачать, а не о её свойствах.",
            f"5. Поэтому точность ~1.0 здесь — тревожный сигнал утечки и перекоса сбора: "
            f"на реальных URL (с путями, без www., с заглавными буквами — в датасете их "
            f"{facts['uppercase_urls']}) такие метрики не повторятся.",
            "6. Сервис получает только сырой URL: HTML-признаки и URLSimilarityIndex в момент "
            "запроса недоступны или считаются по эталонному списку сайтов, а FILENAME — "
            "артефакт сборки датасета.",
            f"7. Дублей URL {facts['duplicate_urls']}: в обучении их убираем, а сплит делаем "
            "по доменам, чтобы один сайт не попадал и в train, и в test.",
        ]
    )


def plot_class_balance(df: pd.DataFrame) -> Figure:
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = _figure()
    bars = ax.bar(
        [CLASSES[c] for c in counts.index], counts, color=[COLORS[c] for c in counts.index]
    )
    ax.bar_label(bars, labels=[f"{n:,}\n{n / len(df):.1%}" for n in counts])
    ax.set_ylabel("URL")
    ax.set_title("Баланс классов")
    ax.margins(y=0.2)
    return fig


def plot_hist_by_class(df: pd.DataFrame, column: str, title: str, clip_quantile: float) -> Figure:
    """Нормированные гистограммы по классам; длинный хвост обрезаем по квантилю."""
    upper = df[column].quantile(clip_quantile)
    bins = np.linspace(df[column].min(), upper, 50)
    fig, ax = _figure()
    for cls, part in df.groupby(TARGET)[column]:
        ax.hist(
            part.clip(upper=upper),
            bins=bins,
            density=True,
            alpha=0.5,
            color=COLORS[cls],
            label=f"{CLASSES[cls]} (медиана {part.median():.2f})",
        )
    ax.set_xlabel(f"{column} (обрезано по {clip_quantile:.0%}-квантилю)")
    ax.set_ylabel("плотность")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_subdomains(df: pd.DataFrame) -> Figure:
    counts = df["NoOfSubDomain"].clip(upper=4)
    share = pd.crosstab(counts, df[TARGET], normalize="columns")
    fig, ax = _figure()
    _grouped_bars(ax, share, [str(i) if i < 4 else "4+" for i in share.index])
    ax.set_xlabel("число поддоменов")
    ax.set_ylabel("доля URL класса")
    ax.set_title("Поддомены по классам")
    return fig


def plot_top_tld(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(10, 5), layout="constrained")
    axes = fig.subplots(1, 2)
    for ax, (cls, part) in zip(axes, df.groupby(TARGET)["TLD"], strict=True):
        top = part.value_counts(normalize=True).head(TOP_TLD)[::-1]
        ax.barh(top.index, top, color=COLORS[cls])
        ax.set_title(f"Топ-{TOP_TLD} TLD: {CLASSES[cls]}")
        ax.set_xlabel("доля URL класса")
    return fig


def plot_url_similarity(df: pd.DataFrame, facts: dict[str, float]) -> Figure:
    fig, ax = _figure()
    bins = np.linspace(0, 100, 51)
    for cls, part in df.groupby(TARGET)["URLSimilarityIndex"]:
        ax.hist(part, bins=bins, density=True, alpha=0.5, color=COLORS[cls], label=CLASSES[cls])
    ax.set_xlabel("URLSimilarityIndex")
    ax.set_ylabel("плотность")
    ax.set_title(
        f"Утечка: =100 у {facts['sim100_legit_share']:.0%} легитимных и "
        f"{facts['sim100_phishing_share']:.1%} фишинга\n"
        f"правило «≠100 → фишинг»: accuracy {facts['sim_rule_accuracy']:.2%}",
        fontsize=10,
    )
    ax.legend()
    return fig


def plot_share_by_category(df: pd.DataFrame, category: pd.Series, title: str) -> Figure:
    """Накопленные столбцы: как распределены категории внутри каждого класса."""
    share = pd.crosstab(df[TARGET], category, normalize="index")
    fig, ax = _figure()
    left = np.zeros(len(share))
    labels = [CLASSES[c] for c in share.index]
    for name in share.columns:
        bars = ax.barh(labels, share[name], left=left, label=name)
        ax.bar_label(
            bars, labels=[f"{v:.0%}" if v >= 0.04 else "" for v in share[name]], label_type="center"
        )
        left += share[name].to_numpy()
    ax.set_xlabel("доля URL класса")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2, fontsize=8)
    return fig


def _grouped_bars(ax: Axes, share: pd.DataFrame, labels: list[str]) -> None:
    x = np.arange(len(share))
    width = 0.4
    for offset, cls in zip((-width / 2, width / 2), share.columns, strict=True):
        ax.bar(x + offset, share[cls], width, color=COLORS[cls], label=CLASSES[cls])
    ax.set_xticks(x, labels)
    ax.legend()


def _figure() -> tuple[Figure, Axes]:
    fig = Figure(figsize=(6, 4.5), layout="constrained")
    return fig, fig.subplots()


def main() -> None:
    setup_mlflow(EXPERIMENT)
    df = load_raw()
    facts = compute_facts(df)
    auc = feature_auc(df)

    with mlflow.start_run(run_name="eda"):
        dataset = mlflow.data.from_pandas(
            df.drop(columns=TARGET), source=DATASET_URL, name="phiusiil-raw", targets="label"
        )
        mlflow.log_input(dataset, context="eda")
        mlflow.log_param("dataset_sha256", CSV_SHA256)
        mlflow.log_metrics(facts)

        figures = {
            "class_balance": plot_class_balance(df),
            "url_length": plot_hist_by_class(df, "URLLength", "Длина URL по классам", 0.99),
            "subdomains": plot_subdomains(df),
            "digit_ratio": plot_hist_by_class(
                df, "DegitRatioInURL", "Доля цифр в URL по классам", 0.99
            ),
            "top_tld": plot_top_tld(df),
            "leak_url_similarity": plot_url_similarity(df, facts),
            "leak_url_prefix": plot_share_by_category(
                df, url_prefix(df["URL"]), "Утечка: начало URL по классам"
            ),
            "leak_url_tail": plot_share_by_category(
                df, url_tail(df["URL"]), "Утечка: что идёт после хоста"
            ),
        }
        for name, fig in figures.items():
            mlflow.log_figure(fig, f"plots/{name}.png")

        mlflow.log_table(auc, "tables/feature_roc_auc.json")
        mlflow.log_table(
            pd.crosstab(
                df["FILENAME"].str.extract(r"^(\D*)", expand=False).replace("", "<число>"),
                df[TARGET].map(CLASSES),
            ).reset_index(names="filename_prefix"),
            "tables/filename_prefix_by_class.json",
        )
        mlflow.log_table(
            df.loc[df["URL"].duplicated(keep=False), ["URL", TARGET]].sort_values("URL"),
            "tables/duplicate_urls.json",
        )
        text = conclusions(facts, auc)
        mlflow.log_text(text, "conclusions.md")
        print(text)


if __name__ == "__main__":
    main()
