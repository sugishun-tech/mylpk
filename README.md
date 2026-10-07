# mylpk / myomo

Cythonで実装した独自LP・MILPソルバー **mylpk** と、Python内で数式を書けるモデリングAPI **myomo** を同梱したプロジェクトです。バージョンは **0.1.0** です。

`solver="mylpk"` は本プロジェクトの改訂単体法と分枝限定法を実行します。SciPyの最適化関数へ処理を転送するラッパーではありません。LU分解・連立方程式の処理にはSciPyのCython LAPACKインターフェースを使用します。`solver="highs"`、`solver="glpk"` は明示的に選ぶ別バックエンドです。

**位置づけ:** 検証可能な実装と数学的な診断機能を備えた初期バージョンです。GLPK/PyomoとのAPI互換、大規模疎問題での性能同等性、あらゆる数理最適化機能、全問題での最速を主張するものではありません。基底分解は密行列LUです。用途への適用前に [制約事項](docs/LIMITATIONS.md) を確認してください。

## インストール

ZIPを展開し、`pyproject.toml` のある `mylpk` ディレクトリで実行します。PyPI公開を前提にした手順ではありません。

```bash
python3.13 -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[dev]'
python examples/production.py
python -m pytest -q
```

CコンパイラとPython開発ヘッダが必要です。Debian系では環境に応じて `build-essential`、使用中のPythonに対応した開発ヘッダ、`python3-venv` を用意してください。数値コアはCythonからCへコンパイルします。本プロジェクト自身にC++ソースはありません。

検証環境は **Linux x86-64 / CPython 3.13.5 / Cython 3.2.4 / NumPy 2.3.5 / SciPy 1.17.0 / GCC 14.2.0** です。依存範囲はPython 3.10以降、SciPy 1.11以上1.18未満ですが、この範囲内の全組合せを検証したわけではありません。SciPy 1.18以降のCython LAPACK整数型・ILP64対応は未対応です。詳細は [ビルド手順](docs/BUILD.md) を参照してください。

## 最初のモデル

```python
from myomo import Model

m = Model("production")
x = m.var("x", kind="integer")
y = m.var("y", kind="integer")

m += 2*x + y <= 14
m += x + 2*y <= 14
m.maximize(3*x + 2*y)

r = m.solve(mip_rel_gap=0).require_optimal()
print(r.status, r[x], r[y], r.objective)
# optimal 5.0 4.0 23.0
```

変数の既定下限は0、既定上限は無限大です。自由変数には `lb=None` を指定します。整数問題と連続問題は変数の定義から判断します。`0 <= x <= 1` の連鎖比較はPythonが途中を真偽値に変換するため使えません。`m += between(0, x, 1)` または変数の `lb` / `ub` を使用します。

## 添字・パラメータ・行列

```python
from myomo import Model
from scipy import sparse

m = Model()
capacity = m.param("capacity", 5)
x = m.vars("x", ["a", "b"], ub=capacity)
m += x.sum() <= capacity
m.maximize(x.dot({"a": 3, "b": 2}))
print(m.solve().require_optimal()[x])

capacity.value = 8
# 更新前の x["a"].value はエラーになります。古い解を現在の解として使いません。
print(m.solve().require_optimal()[x])

bulk = Model()
z = bulk.vars("z", 3)
bulk.add_matrix(sparse.csr_matrix([[1, 1, 1], [1, 0, 0]]), z, ub=[5, 2])
bulk.maximize(z.dot([4, 2, 1]))
print(bulk.solve().require_optimal().objective)  # 14
```

大きな和には `quicksum()`、既存の疎行列には `add_matrix()` または `LinearProblem` を使用します。組込み `sum()` も使えますが、式の繰り返しコピーを発生させます。

## 実装範囲

| 領域 | 同梱機能 |
|---|---|
| 独自LP | 主・双対改訂単体法、二段階法、CSC列アクセス、LU＋eta更新、基底ウォームスタート、退化対策、行・目的係数スケーリング |
| 独自MILP | 最良境界優先探索、擬似コスト分岐、丸め＋固定整数LPヒューリスティック、根ノード二値カバーカット、半連続・半整数変数の再定式化 |
| 再最適化 | `LPSession` によるLU・eta保持、目的係数/RHS更新、独立問題の `solve_many()` |
| 解析 | 元座標の双対値・換算費用・KKT検査、双対問題生成、標準形での基底感度、有理数によるLP最適性証明チェック、IIS、非有界方向 |
| モデリング | 添字付き変数、遅延評価パラメータ、行列一括追加、二次式、論理制約、indicator、区分線形、絶対値、SOS1/SOS2、軟制約、二値×連続積 |
| ワークフロー | 辞書式多目的最適化、二値パターン列挙、パラメータシナリオ、線形モデルの数値スナップショット複製 |
| 連携・入出力 | HiGHS、GLPK、Pyomoブリッジ、独自ソルバー登録、JSON/MPS読書き、LP書出し、highspyによるLP読込み、CLI |

SOS、indicator、区分線形などはLP/MILPへの**再定式化**です。専用のSOS分岐器やネイティブindicator処理ではありません。有限境界が必要な変換では、根拠のない巨大なBig-Mを勝手に設定せず、境界不足をエラーにします。

## 外部ソルバー

```bash
# GLPKのPythonバインディングを追加
python -m pip install '.[glpk]'
python examples/production.py --solver glpk

# LPファイル読込みとPyomoブリッジを追加
python -m pip install '.[interop]'
```

```python
r = m.solve(solver="mylpk")      # 独自エンジン
r = m.solve(solver="highs")      # SciPy同梱HiGHS
r = m.solve(solver="highs-ds")   # 連続LPのみ
r = m.solve(solver="highs-ipm")  # 連続LPのみ
r = m.solve(solver="glpk")       # swiglpkが必要
# 対応する外部ソルバーを別途インストールした場合:
# r = m.solve(solver="pyomo:cbc", backend_options={"seconds": 10})
```

HiGHS経路は実行検証済みです。**GLPK経路、Pyomoブリッジ、highspyによるLP読込みは依存パッケージがこの検証環境に存在せず、実機検証未実施**です。GLPK向け20テストとLP読込み1テストは、必要な依存がある環境で自動的に実行されます。バックエンド間でオプションと返却できる情報は異なります。[API仕様](docs/API.md) を参照してください。

## 最適性と実行可能性

`r.success` は `r.status == "optimal"` のときだけ真です。制限時間終了時の実行可能解や、SLSQPの局所解を大域最適と扱いません。途中解は `r.feasible` と `r.bound` / `r.gap` を確認して使用します。浮動小数点LP/MILPの `optimal` は設定した許容誤差を含む判定です。有理数の厳密な証明チェックとは別です。

連続QP/QCQPは `m.solve(solver="scipy")` を明示するとSLSQPで解きます。返却状態は `locally_optimal` で、`r.success` は偽です。MIQP、MINLP、SDP、SOCP、一般の超越関数NLP、厳密有理数単体法は未実装です。

## ベンチマークと検証

[性能測定](docs/PERFORMANCE.md) に全測定ケースを掲載しています。小規模な密LP・反復更新で短い実行時間が得られた一方、疎LPや整数問題ではHiGHSが高速でした。測定は一つの環境のマイクロベンチマークであり、Pyomoのモデリング速度は未測定です。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/run.py
python tools/validate_wheel.py
python tools/package.py --output ../mylpk-0.1.0.zip
```

テスト結果は [検証報告](reports/VALIDATION.md)、生データは `reports/benchmark.json` と `reports/pytest.xml` にあります。`tools/validate_wheel.py` はクリーンなソースアーカイブからwheelを作り、別のインストール先からテストします。

## ドキュメントと例

[API](docs/API.md)、[数学とアルゴリズム](docs/MATHEMATICS.md)、[ビルド](docs/BUILD.md)、[性能](docs/PERFORMANCE.md)、[制約事項](docs/LIMITATIONS.md)、[参考資料](docs/REFERENCES.md)、[Qiita記事](docs/QIITA.md)。

`examples/` には生産計画、輸送問題、証明・感度・IIS、再最適化、論理・区分線形、多目的・列挙・シナリオ、局所QCQP、行列・ファイル入出力の8例を同梱しています。

`.github`、ライセンスファイル、生成済み拡張バイナリ、ビルドキャッシュは配布ZIPに含めていません。第三者のソルバー本体も同梱せず、外部依存として利用します。
