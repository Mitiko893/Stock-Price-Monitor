"""
良株を見つけるツール - Phase 1〜4
--------------------------------
機能:
  1. watchlist.json に書いた銘柄の株価（現在値・前日比）を取得
  2. PER・PBR・時価総額・決算推移（直近3期）を取得
  3. 各銘柄に関連するニュースを数件取得
  4. ローカルAI（Ollama）でここまでの情報を要約・ニュースの論調を判定
  5. 8つの指標から100点満点でスコアを計算し、良株ランキングを表示
  6. 上位銘柄が一定点数を超えたらデスクトップ通知を出す
  7. 見やすい表形式でターミナルに表示

使い方:
  python main.py
  python main.py --watchlist my_list.json   # 別のリストを使う場合
  python main.py --no-ai                    # AI要約・ニュース判定を使わない場合

必要なライブラリ（初回のみ）:
  pip install yfinance feedparser tabulate requests win10toast

AI要約を使うには（初回のみ）:
  1. https://ollama.com からOllamaをインストール
  2. ターミナルで `ollama pull qwen2.5:7b` を実行してモデルをダウンロード
  3. Ollamaを起動した状態でこのスクリプトを実行

ランキングの配点（合計100点）:
  売上成長率20 / EPS成長率15 / 営業利益率10 / ROE10 / 自己資本比率10 /
  PER10 / 成長見込み10 / ニュース分析10 / チャート5
  ※ 一部の指標（会社計画・業界比較PER）は無料データの制約上、簡易的な代用指標を使っています。
"""

import argparse
import contextlib
import datetime
import html as html_lib
import io
import json
import os
import sys
from urllib.parse import quote

import feedparser
import requests
import yfinance as yf
from tabulate import tabulate

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

# ランキング上位がこの点数以上ならデスクトップ通知を出す(調整したい場合はここを変更)
NOTIFY_THRESHOLD = 70

# バイ&ホールドシミュレーションで検証する年数
BACKTEST_YEARS = [5, 10]

# 移動平均線クロス戦略で使う日数(短期線・長期線)
MA_SHORT_WINDOW = 25
MA_LONG_WINDOW = 75


def load_watchlist(path: str) -> list[dict]:
    """watchlist.json を読み込んで銘柄リストを返す"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["watchlist"]


def get_stock_data(ticker: str) -> dict:
    """
    yfinanceで株価データを取得する。
    ticker例: "7203.T" (東証の場合は末尾に .T を付ける)
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty or len(hist) < 2:
            return {"ticker": ticker, "error": "価格データが取得できませんでした"}

        latest = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2]
        change = latest - prev
        change_pct = (change / prev) * 100

        return {
            "ticker": ticker,
            "price": round(latest, 1),
            "change": round(change, 1),
            "change_pct": round(change_pct, 2),
            "volume": int(hist["Volume"].iloc[-1]),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def get_financial_data(ticker: str) -> dict:
    """
    yfinanceで財務指標を取得する。
      - PER・PBR・時価総額
      - 直近3期分の売上高・純利益の推移（年次）
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        currency = info.get("currency", "JPY")
        sector = info.get("sector")
        industry = info.get("industry")
        per = info.get("trailingPE")
        pbr = info.get("priceToBook")
        market_cap = info.get("marketCap")

        earnings_trend = []
        income_stmt = stock.income_stmt
        if income_stmt is not None and not income_stmt.empty:
            for period in income_stmt.columns[:3]:
                revenue = (
                    income_stmt.loc["Total Revenue", period]
                    if "Total Revenue" in income_stmt.index
                    else None
                )
                net_income = (
                    income_stmt.loc["Net Income", period]
                    if "Net Income" in income_stmt.index
                    else None
                )
                earnings_trend.append({
                    "period": period.strftime("%Y-%m"),
                    "revenue": revenue,
                    "net_income": net_income,
                })

        return {
            "ticker": ticker,
            "currency": currency,
            "sector": sector,
            "industry": industry,
            "per": per,
            "pbr": pbr,
            "market_cap": market_cap,
            "earnings_trend": earnings_trend,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def get_dividend_data(ticker: str, years: int = 10) -> dict:
    """
    配当利回り・配当性向(現在の指標)と、直近N年分の年間配当推移(増配・減配の履歴)を取得する。
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        currency = info.get("currency", "JPY")
        dividend_yield = info.get("dividendYield")
        payout_ratio = info.get("payoutRatio")

        yearly_dividends = []
        dividends = stock.dividends
        if dividends is not None and not dividends.empty:
            by_year = dividends.groupby(dividends.index.year).sum()
            for year, amount in by_year.tail(years).items():
                yearly_dividends.append({"year": int(year), "amount": round(float(amount), 1)})

        return {
            "currency": currency,
            "dividend_yield": dividend_yield,
            "payout_ratio": payout_ratio,
            "yearly_dividends": yearly_dividends,
        }
    except Exception as e:
        return {"error": str(e)}


def get_scoring_metrics(ticker: str) -> dict:
    """
    ランキングに使う指標をまとめて取得する。
      - 売上成長率・EPS成長率（前年同期比、yfinanceの公表値）
      - 営業利益率・ROE
      - 自己資本比率（貸借対照表から計算）
      - PER
      - 成長見込み（アナリスト予想株価と現在株価の差を代用指標にする）
      - チャート（直近終値が25日移動平均線より上かどうか）
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        revenue_growth = info.get("revenueGrowth")
        earnings_growth = info.get("earningsGrowth")
        operating_margin = info.get("operatingMargins")
        roe = info.get("returnOnEquity")
        per = info.get("trailingPE")

        # 自己資本比率 = 自己資本 ÷ 総資産
        equity_ratio = None
        balance_sheet = stock.balance_sheet
        if balance_sheet is not None and not balance_sheet.empty:
            latest_period = balance_sheet.columns[0]
            equity = None
            for key in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if key in balance_sheet.index:
                    equity = balance_sheet.loc[key, latest_period]
                    break
            assets = (
                balance_sheet.loc["Total Assets", latest_period]
                if "Total Assets" in balance_sheet.index
                else None
            )
            if equity is not None and assets:
                equity_ratio = equity / assets

        # 成長見込み(代用指標): アナリスト予想の平均株価と現在株価の差
        expected_upside = None
        current_price = info.get("currentPrice")
        target_price = info.get("targetMeanPrice")
        if current_price and target_price:
            expected_upside = (target_price - current_price) / current_price

        # チャート: 直近終値が25日移動平均線より上にあるか
        chart_uptrend = None
        hist = stock.history(period="3mo")
        if not hist.empty and len(hist) >= 25:
            ma25 = hist["Close"].rolling(25).mean().iloc[-1]
            chart_uptrend = hist["Close"].iloc[-1] > ma25

        return {
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "operating_margin": operating_margin,
            "roe": roe,
            "equity_ratio": equity_ratio,
            "per": per,
            "expected_upside": expected_upside,
            "chart_uptrend": chart_uptrend,
        }
    except Exception as e:
        return {"error": str(e)}


def score_threshold(value, thresholds: list) -> int:
    """
    value が None なら0点。
    thresholds は [(しきい値, 得点), ...] を大きい順に並べたリスト。
    value が各しきい値以上なら、最初に条件を満たした得点を返す。
    """
    if value is None:
        return 0
    for threshold, points in thresholds:
        if value >= threshold:
            return points
    return 0


def score_per(per) -> int:
    """PERは低いほど割安とみなし、高得点にする(簡易的な絶対基準)"""
    if per is None or per <= 0:
        return 0
    if per <= 10:
        return 10
    if per <= 15:
        return 8
    if per <= 20:
        return 6
    if per <= 30:
        return 4
    return 2


def get_historical_return(ticker: str, years: int):
    """
    指定した年数前から現在までの株価騰落率(%)を計算する。
    データが取れない場合は None を返す。
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{years}y")
        if hist.empty or len(hist) < 2:
            return None
        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        return (end_price - start_price) / start_price * 100
    except Exception:
        return None


def backtest_buy_and_hold(ticker: str, years: int) -> dict:
    """
    指定した年数前に一括投資し、そのまま現在まで保有した場合のシミュレーション。
    値上がり益 + 保有期間中に受け取った配当金の合計で「トータルリターン」を計算する
    (配当を再投資はせず、現金として受け取った前提のシンプルな計算)。
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{years}y", auto_adjust=False)
        if hist.empty or len(hist) < 2:
            return {"error": "データが取得できませんでした"}

        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        total_dividends = hist["Dividends"].sum() if "Dividends" in hist.columns else 0

        total_return_pct = ((end_price - start_price) + total_dividends) / start_price * 100

        # トータルリターンをもとに年率換算(複利)する
        annualized_pct = None
        total_return_ratio = 1 + (total_return_pct / 100)
        if total_return_ratio > 0:
            annualized_pct = (total_return_ratio ** (1 / years) - 1) * 100

        return {
            "start_price": start_price,
            "end_price": end_price,
            "total_dividends": total_dividends,
            "total_return_pct": total_return_pct,
            "annualized_pct": annualized_pct,
        }
    except Exception as e:
        return {"error": str(e)}


def backtest_ma_crossover(ticker: str, short_window: int = 25, long_window: int = 75, years: int = 10) -> dict:
    """
    移動平均線クロス戦略のシミュレーション。
    短期線が長期線を上に抜けたら「買い」(ゴールデンクロス)、
    下に抜けたら「売り」(デッドクロス)として、シグナル通りに全額売買した場合の結果を計算する。
    (売買手数料・税金は考慮していない簡易シミュレーション)
    """
    try:
        stock = yf.Ticker(ticker)
        # 移動平均の計算に必要な分、長めにデータを取得しておく
        hist = stock.history(period=f"{years + 1}y", auto_adjust=False)
        if hist.empty or len(hist) < long_window + 10:
            return {"error": "データが不足しています"}

        hist["MA_short"] = hist["Close"].rolling(short_window).mean()
        hist["MA_long"] = hist["Close"].rolling(long_window).mean()
        hist = hist.dropna(subset=["MA_short", "MA_long"])
        if hist.empty:
            return {"error": "移動平均を計算できませんでした"}

        initial_capital = 1_000_000  # 仮の初期資金(100万円)
        cash = initial_capital
        shares = 0
        trade_count = 0
        position = "none"
        prev_diff = None

        for _, row in hist.iterrows():
            diff = row["MA_short"] - row["MA_long"]
            price = row["Close"]

            if prev_diff is not None:
                if prev_diff <= 0 and diff > 0 and position == "none":
                    # ゴールデンクロス: 全額買う
                    shares = cash / price
                    cash = 0
                    position = "long"
                    trade_count += 1
                elif prev_diff >= 0 and diff < 0 and position == "long":
                    # デッドクロス: 全部売る
                    cash = shares * price
                    shares = 0
                    position = "none"
                    trade_count += 1

            prev_diff = diff

        last_price = hist["Close"].iloc[-1]
        final_value = cash + shares * last_price
        strategy_return_pct = (final_value - initial_capital) / initial_capital * 100

        start_price = hist["Close"].iloc[0]
        buy_hold_return_pct = (last_price - start_price) / start_price * 100

        return {
            "strategy_return_pct": strategy_return_pct,
            "buy_hold_return_pct": buy_hold_return_pct,
            "trade_count": trade_count,
        }
    except Exception as e:
        return {"error": str(e)}


def compute_score(metrics: dict, sentiment: str) -> dict:
    """8つの指標(+ニュース分析)から100点満点のスコアを計算する"""
    breakdown = {
        "売上成長率": score_threshold(metrics.get("revenue_growth"), [(0.15, 20), (0.10, 15), (0.05, 10), (0.0, 5)]),
        "EPS成長率": score_threshold(metrics.get("earnings_growth"), [(0.15, 15), (0.10, 11), (0.05, 7), (0.0, 3)]),
        "営業利益率": score_threshold(metrics.get("operating_margin"), [(0.15, 10), (0.10, 7), (0.05, 4), (0.0, 2)]),
        "ROE": score_threshold(metrics.get("roe"), [(0.15, 10), (0.10, 7), (0.05, 4), (0.0, 2)]),
        "自己資本比率": score_threshold(metrics.get("equity_ratio"), [(0.50, 10), (0.40, 8), (0.30, 6), (0.20, 4), (0.0, 2)]),
        "PER": score_per(metrics.get("per")),
        "成長見込み": score_threshold(metrics.get("expected_upside"), [(0.20, 10), (0.10, 7), (0.0, 4)]),
        "ニュース分析": {"ポジティブ": 10, "ニュートラル": 5, "ネガティブ": 0}.get(sentiment, 5),
        "チャート": 5 if metrics.get("chart_uptrend") else 0,
    }
    total = sum(breakdown.values())
    return {"breakdown": breakdown, "total": total}


def compute_sector_peers(companies: list) -> dict:
    """
    watchlist内で同じセクター(業種)の銘柄同士をグループ化し、平均値を計算する。
    ※ 本当の「業界全体の平均」ではなく、あくまでこのwatchlistに入っている
      同業種銘柄同士の平均であることに注意(無料データの制約のため)。
    """
    groups: dict = {}
    for c in companies:
        sector = c["financial_data"].get("sector")
        if not sector:
            continue
        groups.setdefault(sector, []).append(c)

    def avg(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    sector_stats = {}
    for sector, members in groups.items():
        sector_stats[sector] = {
            "count": len(members),
            "per": avg([m["financial_data"].get("per") for m in members]),
            "pbr": avg([m["financial_data"].get("pbr") for m in members]),
            "roe": avg([m["scoring_metrics"].get("roe") for m in members]),
            "operating_margin": avg([m["scoring_metrics"].get("operating_margin") for m in members]),
        }
    return sector_stats


def format_large_amount(value, currency: str = "JPY") -> str:
    """
    大きな金額を通貨に応じた単位で読みやすく表示する。
    日本円は「億円」単位、米ドルは「B(10億ドル)」単位、それ以外は通貨コード付きの生の数値。
    データがなければ '-' を返す。
    """
    if value is None:
        return "-"
    if currency == "JPY":
        return f"{value / 1e8:,.0f}億円"
    if currency == "USD":
        return f"${value / 1e9:,.1f}B"
    return f"{value:,.0f} {currency}"


def format_price(value, currency: str = "JPY") -> str:
    """株価などの単価を通貨に応じて表示する。データがなければ '-' を返す"""
    if value is None:
        return "-"
    if currency == "JPY":
        return f"{value:,.1f}円"
    if currency == "USD":
        return f"${value:,.2f}"
    return f"{value:,.2f} {currency}"


def format_per_share(value, currency: str = "JPY") -> str:
    """1株あたりの配当などを通貨に応じて表示する。データがなければ '-' を返す"""
    if value is None:
        return "-"
    if currency == "JPY":
        return f"{value:,.1f}円/株"
    if currency == "USD":
        return f"${value:,.2f}/株"
    return f"{value:,.2f} {currency}/株"


def format_ratio(value) -> str:
    """PER・PBRなどの倍率を小数点1桁の文字列に変換する。データがなければ '-' を返す"""
    if value is None:
        return "-"
    return f"{value:.1f}倍"


def get_news(company_name: str, max_items: int = 3) -> list:
    """Google News RSS で会社名に関するニュースを取得する。無料でAPIキー不要"""
    query = quote(f"{company_name} 株価")
    url = f"https://news.google.com/rss/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"

    try:
        feed = feedparser.parse(url)
        news_items = []
        for entry in feed.entries[:max_items]:
            news_items.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published", "不明"),
            })
        return news_items
    except Exception as e:
        return [{"title": f"ニュース取得エラー: {e}", "link": "", "published": ""}]


def build_analysis_prompt(name: str, stock_data: dict, financial_data: dict, news_items: list) -> str:
    """AIに渡すプロンプトを組み立てる(1行目:ニュースの論調、2行目以降:要約)"""
    lines = [f"以下は「{name}」という会社の株価・財務データ・ニュース見出しです。"]
    currency = financial_data.get("currency", "JPY")

    if "error" not in stock_data:
        sign = "+" if stock_data["change"] >= 0 else ""
        lines.append(f"株価: {format_price(stock_data['price'], currency)} (前日比 {sign}{stock_data['change_pct']}%)")

    if "error" not in financial_data:
        lines.append(
            f"PER: {format_ratio(financial_data.get('per'))} / "
            f"PBR: {format_ratio(financial_data.get('pbr'))} / "
            f"時価総額: {format_large_amount(financial_data.get('market_cap'), currency)}"
        )
        earnings_trend = financial_data.get("earnings_trend", [])
        if earnings_trend:
            trend_text = " / ".join(
                f"{e['period']}期 売上{format_large_amount(e['revenue'], currency)}"
                f"・純利益{format_large_amount(e['net_income'], currency)}"
                for e in earnings_trend
            )
            lines.append(f"決算推移: {trend_text}")

    if news_items:
        lines.append("最近のニュース見出し:")
        for news in news_items:
            lines.append(f"- {news['title']}")

    lines.append(
        "\n1行目に、ニュース全体の論調を「ポジティブ」「ニュートラル」「ネガティブ」のいずれか一語だけで書いてください。"
        "2行目以降に、上記のデータだけをもとに日本語で3行以内の客観的な要約を書いてください。"
        "投資判断や「買い」「売り」の推奨は書かず、数値やニュースから読み取れる事実の傾向だけを述べてください。"
        "書き直しや言い訳、英語の説明文は一切含めないでください。"
    )
    return "\n".join(lines)


def get_ai_analysis(name: str, stock_data: dict, financial_data: dict, news_items: list) -> dict:
    """
    ローカルのOllamaにプロンプトを送り、「ニュースの論調」と「要約」を取得する。
    Ollamaが起動していない場合はニュートラル扱いにする。
    """
    prompt = build_analysis_prompt(name, stock_data, financial_data, news_items)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=60,
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip()

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return {"sentiment": "ニュートラル", "summary": "(要約を取得できませんでした)"}

        sentiment = "ニュートラル"
        for word in ["ポジティブ", "ネガティブ", "ニュートラル"]:
            if word in lines[0]:
                sentiment = word
                break
        summary = "\n".join(lines[1:]) if len(lines) > 1 else lines[0]
        return {"sentiment": sentiment, "summary": summary}
    except requests.exceptions.ConnectionError:
        return {"sentiment": "ニュートラル", "summary": "(Ollamaに接続できません。Ollamaを起動しているか確認してください)"}
    except Exception as e:
        return {"sentiment": "ニュートラル", "summary": f"(AI要約でエラーが発生しました: {e})"}


def send_notification(title: str, message: str) -> None:
    """デスクトップに通知を表示する(Windows向け)。ライブラリが無ければ静かにスキップする"""
    try:
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(title, message, duration=10, threaded=True)
    except ImportError:
        print("(デスクトップ通知を使うには 'pip install win10toast' が必要です)")
    except Exception as e:
        print(f"(通知の送信に失敗しました: {e})")


def print_report(watchlist: list, use_ai: bool = True) -> None:
    """全銘柄の株価・財務データ・ニュース・AI要約・ランキングをまとめて表示する"""
    companies = []

    for item in watchlist:
        ticker = item["ticker"]
        name = item["name"]

        stock_data = get_stock_data(ticker)
        financial_data = get_financial_data(ticker)
        dividend_data = get_dividend_data(ticker)
        news_items = get_news(name)
        scoring_metrics = get_scoring_metrics(ticker)

        if use_ai:
            ai_analysis = get_ai_analysis(name, stock_data, financial_data, news_items)
        else:
            ai_analysis = {"sentiment": "ニュートラル", "summary": None}

        score = compute_score(scoring_metrics, ai_analysis["sentiment"])

        companies.append({
            "name": name,
            "ticker": ticker,
            "stock_data": stock_data,
            "financial_data": financial_data,
            "dividend_data": dividend_data,
            "news_items": news_items,
            "ai_analysis": ai_analysis,
            "scoring_metrics": scoring_metrics,
            "score": score,
        })

    print("\n=== 株価一覧 ===")
    table_rows = []
    for c in companies:
        stock_data = c["stock_data"]
        financial_data = c["financial_data"]
        currency = financial_data.get("currency", "JPY")
        if "error" in stock_data:
            table_rows.append([c["name"], c["ticker"], "取得失敗", "-", "-", "-", "-", "-"])
        else:
            sign = "+" if stock_data["change"] >= 0 else ""
            change_str = format_price(stock_data["change"], currency)
            table_rows.append([
                c["name"],
                c["ticker"],
                format_price(stock_data["price"], currency),
                f"{sign}{change_str} ({sign}{stock_data['change_pct']}%)",
                f"{stock_data['volume']:,}",
                format_ratio(financial_data.get("per")),
                format_ratio(financial_data.get("pbr")),
                format_large_amount(financial_data.get("market_cap"), currency),
            ])
    print(tabulate(
        table_rows,
        headers=["銘柄名", "コード", "株価", "前日比", "出来高", "PER", "PBR", "時価総額"],
        tablefmt="simple",
    ))

    print("\n=== 決算推移（直近3期・年次） ===")
    for c in companies:
        print(f"\n【{c['name']}】")
        earnings_trend = c["financial_data"].get("earnings_trend", [])
        if not earnings_trend:
            print("  (決算データが取得できませんでした)")
            continue
        currency = c["financial_data"].get("currency", "JPY")
        earnings_rows = [
            [e["period"], format_large_amount(e["revenue"], currency), format_large_amount(e["net_income"], currency)]
            for e in earnings_trend
        ]
        print(tabulate(earnings_rows, headers=["決算期", "売上高", "純利益"], tablefmt="simple"))

    print("\n=== 配当分析 ===")
    for c in companies:
        print(f"\n【{c['name']}】")
        dividend_data = c["dividend_data"]
        if "error" in dividend_data:
            print("  (配当データを取得できませんでした)")
            continue

        yield_value = dividend_data.get("dividend_yield")
        payout_value = dividend_data.get("payout_ratio")
        currency = dividend_data.get("currency", "JPY")
        # 注意: yfinanceのdividendYieldは既に「%表記の数値」(例: 3.26 = 3.26%)で返ってくるため、
        # 100倍しない。payoutRatioは従来通り「割合」(例: 0.322 = 32.2%)で返るため100倍する。
        yield_text = f"{yield_value:.2f}%" if yield_value else "-"
        payout_text = f"{payout_value * 100:.1f}%" if payout_value else "-"
        print(f"  配当利回り: {yield_text} / 配当性向: {payout_text}")

        yearly_dividends = dividend_data.get("yearly_dividends", [])
        if not yearly_dividends:
            print("  (配当の支払い履歴がありません)")
            continue

        dividend_rows = []
        prev_amount = None
        for item in yearly_dividends:
            if prev_amount is None:
                trend = "-"
            elif item["amount"] > prev_amount:
                trend = "増配"
            elif item["amount"] < prev_amount:
                trend = "減配"
            else:
                trend = "据え置き"
            dividend_rows.append([item["year"], format_per_share(item["amount"], currency), trend])
            prev_amount = item["amount"]
        print(tabulate(dividend_rows, headers=["年", "年間配当(1株)", "前年比"], tablefmt="simple"))

    print("\n=== 関連ニュース ===")
    for c in companies:
        print(f"\n【{c['name']}】")
        if not c["news_items"]:
            print("  (ニュースが見つかりませんでした)")
        for news in c["news_items"]:
            print(f"  - {news['title']}")
            if news["link"]:
                print(f"    {news['link']}")

    if use_ai:
        print("\n=== AI要約 ===")
        for c in companies:
            print(f"\n【{c['name']}】(ニュース論調: {c['ai_analysis']['sentiment']})")
            print(f"  {c['ai_analysis']['summary']}")

    print("\n=== 良株ランキング ===")
    ranked = sorted(companies, key=lambda c: c["score"]["total"], reverse=True)
    ranking_rows = [
        [i + 1, c["name"], f"{c['score']['total']}点"]
        for i, c in enumerate(ranked)
    ]
    print(tabulate(ranking_rows, headers=["順位", "銘柄名", "合計点(100点満点)"], tablefmt="simple"))

    print("\n--- 内訳 ---")
    for c in ranked:
        breakdown_text = " / ".join(f"{k}{v}点" for k, v in c["score"]["breakdown"].items())
        print(f"【{c['name']}】{breakdown_text}")

    print("\n=== 業界内比較（このwatchlist内の同業種銘柄同士） ===")
    print("※ 市場全体の業界平均ではなく、watchlistに登録している同業種の銘柄同士を比較した平均値です。")
    print("  (無料データの制約上、市場全体の業界平均は取得していません)")
    sector_stats = compute_sector_peers(companies)
    for c in ranked:
        sector = c["financial_data"].get("sector")
        print(f"\n【{c['name']}】業種: {sector or '不明'}")
        if not sector or sector_stats.get(sector, {}).get("count", 0) < 2:
            print("  (同業種の比較対象がwatchlist内にありません)")
            continue

        stats = sector_stats[sector]
        compare_rows = [
            ["PER", format_ratio(c["financial_data"].get("per")), format_ratio(stats["per"])],
            ["PBR", format_ratio(c["financial_data"].get("pbr")), format_ratio(stats["pbr"])],
            [
                "ROE",
                f"{c['scoring_metrics'].get('roe') * 100:.1f}%" if c["scoring_metrics"].get("roe") is not None else "-",
                f"{stats['roe'] * 100:.1f}%" if stats["roe"] is not None else "-",
            ],
            [
                "営業利益率",
                f"{c['scoring_metrics'].get('operating_margin') * 100:.1f}%" if c["scoring_metrics"].get("operating_margin") is not None else "-",
                f"{stats['operating_margin'] * 100:.1f}%" if stats["operating_margin"] is not None else "-",
            ],
        ]
        print(tabulate(
            compare_rows,
            headers=["指標", "自社", f"同業種平均(n={stats['count']})"],
            tablefmt="simple",
        ))

    print("\n=== スコア検証（簡易・過去の株価騰落率との比較） ===")
    print("※ スコアは「現在の財務状況」の点数です。過去の値動きと比べることで、")
    print("  スコアが高い銘柄が実際に値上がりしてきたかの参考になります(銘柄数が少ないと参考程度です)。")
    verify_rows = []
    for c in ranked:
        return_1y = get_historical_return(c["ticker"], 1)
        return_3y = get_historical_return(c["ticker"], 3)
        verify_rows.append([
            c["name"],
            f"{c['score']['total']}点",
            f"{return_1y:+.1f}%" if return_1y is not None else "-",
            f"{return_3y:+.1f}%" if return_3y is not None else "-",
        ])
    print(tabulate(verify_rows, headers=["銘柄名", "スコア", "1年騰落率", "3年騰落率"], tablefmt="simple"))

    print("\n=== バイ&ホールド シミュレーション（一括投資・配当込み） ===")
    print("※ 過去の実績であり、将来の成果を保証するものではありません。")
    for years in BACKTEST_YEARS:
        print(f"\n--- {years}年前に一括投資して保有し続けた場合 ---")
        bh_rows = []
        for c in ranked:
            result = backtest_buy_and_hold(c["ticker"], years)
            currency = c["financial_data"].get("currency", "JPY")
            if "error" in result:
                bh_rows.append([c["name"], "-", "-", "-"])
            else:
                annualized = (
                    f"{result['annualized_pct']:+.1f}%"
                    if result["annualized_pct"] is not None
                    else "-"
                )
                bh_rows.append([
                    c["name"],
                    f"{result['total_return_pct']:+.1f}%",
                    annualized,
                    format_per_share(result["total_dividends"], currency),
                ])
        print(tabulate(
            bh_rows,
            headers=["銘柄名", "トータルリターン", "年率換算", "累計配当(1株)"],
            tablefmt="simple",
        ))

    print(f"\n=== テクニカル戦略バックテスト（{MA_SHORT_WINDOW}日線・{MA_LONG_WINDOW}日線クロス） ===")
    print("※ 過去の実績であり、将来の成果を保証するものではありません。売買手数料・税金は考慮していません。")
    tech_rows = []
    for c in ranked:
        result = backtest_ma_crossover(c["ticker"], MA_SHORT_WINDOW, MA_LONG_WINDOW, years=10)
        if "error" in result:
            tech_rows.append([c["name"], "-", "-", "-"])
        else:
            tech_rows.append([
                c["name"],
                f"{result['strategy_return_pct']:+.1f}%",
                f"{result['buy_hold_return_pct']:+.1f}%",
                f"{result['trade_count']}回",
            ])
    print(tabulate(
        tech_rows,
        headers=["銘柄名", "戦略リターン(10年)", "買い持ちのみ", "売買回数"],
        tablefmt="simple",
    ))

    # 上位銘柄が基準点を超えていたらデスクトップ通知
    if ranked and ranked[0]["score"]["total"] >= NOTIFY_THRESHOLD:
        top = ranked[0]
        send_notification(
            "良株ランキング通知",
            f"{top['name']}が{top['score']['total']}点でトップです",
        )


class _Tee:
    """複数の出力先(通常のターミナル + メモリ上のバッファ)に同時に書き込むためのヘルパー"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def save_html_report(report_text: str, output_path: str) -> None:
    """ターミナル出力のテキストを、スマホでも見やすいHTMLファイルとして保存する"""
    escaped_text = html_lib.escape(report_text)
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    page = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>良株スクリーナー レポート</title>
<style>
  body {{
    background: #0f172a;
    color: #e2e8f0;
    font-family: "Hiragino Sans", "Meiryo", monospace;
    padding: 16px;
    margin: 0;
  }}
  h1 {{ font-size: 1.1rem; color: #93c5fd; }}
  .updated {{ color: #94a3b8; font-size: 0.85rem; margin-bottom: 16px; }}
  pre {{ white-space: pre-wrap; word-break: break-word; font-size: 0.8rem; line-height: 1.5; }}
</style>
</head>
<body>
<h1>良株スクリーナー レポート</h1>
<div class="updated">最終更新: {generated_at}</div>
<pre>{escaped_text}</pre>
</body>
</html>
"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)


def main():
    parser = argparse.ArgumentParser(description="株価とニュースを表示するツール")
    parser.add_argument(
        "--watchlist",
        default="watchlist.json",
        help="監視銘柄リストのJSONファイル (デフォルト: watchlist.json)",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="AI要約・ニュース判定(Ollama)を使わずに実行する",
    )
    parser.add_argument(
        "--html",
        metavar="PATH",
        help="結果をHTMLファイルとして保存する(例: --html docs/index.html)",
    )
    args = parser.parse_args()

    try:
        watchlist = load_watchlist(args.watchlist)
    except FileNotFoundError:
        print(f"エラー: {args.watchlist} が見つかりません。")
        sys.exit(1)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"エラー: watchlist.json の形式が正しくありません ({e})")
        sys.exit(1)

    if args.html:
        # ターミナルに表示しつつ、同時にHTML用のテキストとしても記録する(2重実行を避けるため)
        buffer = io.StringIO()
        tee = _Tee(sys.stdout, buffer)
        with contextlib.redirect_stdout(tee):
            print_report(watchlist, use_ai=not args.no_ai)
        save_html_report(buffer.getvalue(), args.html)
        print(f"\n(HTMLレポートを {args.html} に保存しました)")
    else:
        print_report(watchlist, use_ai=not args.no_ai)


if __name__ == "__main__":
    main()
