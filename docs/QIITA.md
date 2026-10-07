# CythonでLP/MILPソルバー「mylpk」とモデリングAPI「myomo」を実装する

タグ候補: Python / Cython / 数理最適化 / 線形計画法 / 整数計画法

## 何を作ったか

線形計画問題と混合整数線形計画問題を扱うソルバー **mylpk** と、その問題をPythonの数式で記述する **myomo** を、一つのプロジェクトにまとめた。

独自の改訂単体法をCythonで実装し、整数問題はそのLPエンジンを使う分枝限定法で解く。myomoからは、独自エンジンに加えてHiGHSやGLPKへ明示的に切り替えられる。基底の再利用、双対値、感度分析、IIS、有理数による証明チェックも同梱した。

今回のバージョンは0.1.0である。産業用ソルバーと同じ成熟度、GLPK/Pyomoとの完全互換、全問題での最速、あらゆる数学的機能の実装完了を宣言するものではない。動作範囲と、測定で不利になったケースも掲載する。

## 1. まず整数問題を解く

ソースを展開し、`pyproject.toml` のあるディレクトリでインストールする。PyPIへの公開を前提にせず、ローカルソースを指定する。

```bash
python3.13 -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[dev]'
```

Cコンパイラと、使用するPythonに対応した開発ヘッダが必要になる。今回の検証環境はCPython 3.13.5、Cython 3.2.4、NumPy 2.3.5、SciPy 1.17.0、GCC 14.2.0、Linux x86-64である。

次の問題を考える。

$$
\begin{aligned}
\max\quad &3x+2y\\
\text{subject to}\quad &2x+y\le14,\\
&x+2y\le14,\\
&x,y\in\mathbb Z_{\ge0}.
\end{aligned}
$$

myomoではそのまま記述できる。

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
```

実行結果は次のとおり。

```text
optimal 5.0 4.0 23.0
```

変数の既定下限は0、上限は無限大である。自由変数は `lb=None`、二値変数は `kind="binary"` とする。結果を読む前に状態を確認し、最適解が必要なら `require_optimal()` を使う。

## 2. myomoの書きやすさをどう設計したか

myomoはPythonの内部DSLとして作った。変数生成、制約追加、目的設定、求解を直接実行できるようにし、小さなモデルでは専用の継承クラスやデコレータを必要としない設計にした。

「Pyomoより誰にとっても使いやすい」という主張はしていない。UIの使いやすさと、対応範囲やモデリング速度は別に評価すべきだからだ。ここで狙ったのは、小規模から中規模のLP/MILPで記述の準備を減らすことである。Pyomoの基本的な記述例は[公式チュートリアル](https://pyomo.readthedocs.io/en/stable/getting_started/pyomo_overview/simple_examples.html)を参照できる。

添字付き変数は、名前付き集合やその直積から作れる。

```python
from myomo import Model

m = Model("transport")
x = m.vars("ship", ["a", "b"], ["c", "d"], ub=10)

m += x.sum("a", "*") <= 3
m += x.sum("b", "*") <= 4
m += x.sum("*", "c") >= 2
m += x.sum("*", "d") >= 4

m.minimize(x.dot({
    ("a", "c"): 1,
    ("a", "d"): 3,
    ("b", "c"): 2,
    ("b", "d"): 1,
}))

r = m.solve().require_optimal()
print(r.objective)  # 6.0
print(r[x])
```

`x.sum("a", "*")` は、第一添字をaに固定した和である。辞書の費用は添字で対応させる。IndexedVarsを反復すると変数が得られ、キーが必要な場合は `.keys()` を使う。

一方、Pythonの `0 <= x <= 1` は連鎖比較であり、途中で真偽値判定が入る。myomoはこれを黙って誤解釈せずに拒否する。範囲制約には `between(0, x, 1)`、または変数の上下限を使う。

## 3. パラメータ更新後の古い解を使わない

```python
from myomo import Model

m = Model()
capacity = m.param("capacity", 5)
x = m.var("x", ub=capacity)
m.maximize(2*x)

first = m.solve().require_optimal()
print(first.objective)  # 10.0

capacity.value = 8
second = m.solve().require_optimal()
print(second.objective)  # 16.0
```

パラメータは係数や境界に遅延評価で組み込む。モデルを更新すると、以前の `x.value` は現在のモデルに対応しなくなるため無効化する。

`first[x]` のように結果オブジェクトから読む値は、過去の求解のスナップショットとして保持される。過去の解と現在のモデル状態を区別するための設計である。

大きな和には `quicksum()` を用意した。係数辞書を追加のたびにコピーする代わりに、一回の走査で蓄積する。すでに行列がある場合には式を一項ずつ作らず、`add_matrix()` または `LinearProblem` に渡せる。

## 4. mylpkの数値コア

独自LPエンジンでは、次の標準形を扱う。

$$
\min c^Tx,\qquad Ax=b,\qquad x\ge0.
$$

入力の自由変数、上下限、両側制約は、シフト・変数分割・スラック追加によって変換する。変換の対応を保存し、解と双対値を元の問題へ戻す。

改訂単体法では、基底行列Bを使って

$$
x_B=B^{-1}b,\qquad B^T\pi=c_B,\qquad r_j=c_j-a_j^T\pi
$$

を計算する。換算費用と実行可能性を見ながら基底を交換する。主単体法と双対単体法を実装し、初期実行可能基底の構成には二段階法を用いた。

毎回逆行列を計算する方式は避け、LU分解とeta更新を保持する。入力行列はCSCで保持し、列方向に非ゼロ要素を走査する。数値ピボットのループはCythonの型付きmemoryviewを使い、`nogil` で実行する。memoryviewによるバッファアクセスとGILの扱いは[Cythonの公式文書](https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html)に説明がある。

LAPACKへの呼出しは、SciPyのCythonインターフェースを使う。

```cython
from scipy.linalg.cython_lapack cimport dgetrf, dgetrs
```

ここで利用しているのはLU分解・連立方程式の処理である。`solver="mylpk"` のLPを `scipy.optimize.linprog` に渡す構成ではない。数値基盤としてのLAPACK利用と、最適化アルゴリズム自体の外部委譲は区別している。[SciPyのCython LAPACK文書](https://docs.scipy.org/doc/scipy/reference/linalg.cython_lapack.html)がインターフェースの参照先になる。

この版のLAPACK整数は32ビットである。依存は `scipy>=1.11,<1.18` に制限し、SciPy 1.18以降で文書化されている新しい整数型・ILP64対応は未対応としている。試していないABIに対して互換性を宣言しない。

重要な制約として、**入力行列は疎形式でも、基底LUは密行列**である。標準形の行数mに対し、基底の保存量はO(m²)、再分解はO(m³)になる。大規模疎LPまでこの構成で押し切れるわけではない。

## 5. 整数問題は独自LPによる分枝限定法

MILPでは整数条件を緩和したLPを解き、その目的値を探索の境界として使う。非整数の変数を選び、床・天井で二つの領域に分ける。

$$
x_j\le\lfloor x_j^*\rfloor\quad\text{または}\quad x_j\ge\lceil x_j^*\rceil.
$$

探索には最良境界優先のヒープと擬似コストを使い、整数変数を丸めて固定したLPから暫定解を探索する。根ノードでは、二値ナップサック型の行に対して限定的なカバーカットを追加する。

この探索管理部分はPythonで、各LPの数値ピボットがCythonで動く。全処理がCythonという実装ではない。また、Gomory/MIRカット、強分枝、一般のlazy constraint、並列MILP探索などは未実装である。

LP緩和の非有界性にも注意した。LPが非有界でも、整数点を持たない場合にはMILP自体は実行不可能になり得る。この情報だけでMILPを非有界と断定せず、`relaxation_unbounded` を返す。

## 6. 再最適化ではLUを残す

目的係数だけを繰り返し更新する場面で、毎回モデル変換からLU分解までやり直す必要はない。`LPSession` は独自ソルバーのワークスペースを保持する。

```python
from mylpk import LinearProblem, LPSession

p = LinearProblem(
    c=[-3, -2],
    A=[[1, 1], [1, 0], [0, 1]],
    row_upper=[4, 2, 3],
)

session = LPSession(p)
print(session.last_result.objective)  # -10.0

r = session.resolve(c=[-2, -4]).require_optimal()
print(r.objective)                    # -14.0
print(r.raw["persistent_workspace"])  # True

r = session.resolve(row_upper=[3, 2, 3]).require_optimal()
```

行列と変数境界は固定し、目的係数、目的の向き・定数、行の境界値を変更する。等式/不等式や有限/無限といった行の構造は変えない。両方の実行可能性が崩れた場合には、独自二段階法で再開する。

`m.solve(warm_start=True)` は前回の基底を渡す便利機能で、LUそのものを保持するLPSessionとは異なる。再利用する情報の範囲を分けている。

## 7. 双対値から有理数の証明チェックまで

数値を返すだけでなく、その解を調べる機能を追加した。

```python
from mylpk import (
    LinearProblem, solve, dual_problem,
    make_exact_certificate, basis_sensitivity,
)

p = LinearProblem(
    c=[5, 4], A=[[3, 2]], row_upper=18,
    upper=[7, 8], sense="max", offset=3,
)

r = solve(p).require_optimal()
dual = solve(dual_problem(p)).require_optimal()
print(r.objective, dual.objective)

proof = make_exact_certificate(p, r)
print(proof["certified"])         # True
print(proof["primal_objective"])  # 115/3

ranges = basis_sensitivity(p, r)
print(ranges["coordinate_system"])  # scaled_standard_form
```

`make_exact_certificate()` は、浮動小数点で求めた主解と双対乗数を有理数へ再構成し、主実行可能性、乗数の符号、停留性、主双対目的値の一致をFractionで検査する。再構成や検査が失敗した場合は `certified=False` を返す。

これは厳密有理数単体法ではない。対象の入力は、**保存済みfloat64を十進文字列にしたものを厳密な有理数として解釈した問題**である。入力前に丸められた値を自動的に元の分数へ戻すわけでもない。

感度分析にも座標系の区別がある。この版の区間は、スケーリング済み標準形で一つの係数だけを変え、同じ基底が最適である範囲を計算したものだ。元の係数での区間や、複数係数の同時変動領域ではない。

矛盾した制約の診断には `find_iis()` を使う。削除法による包含関係で極小な矛盾集合を求める。最小要素数の集合ではなく、既定では整数性を緩和する。ソルバーの制限に達した場合は、極小性を確認できたことにしない。

## 8. 論理制約と区分線形を短く書く

```python
from myomo import Model

m = Model()
on = m.var("on", kind="binary")
amount = m.var("amount", ub=3)

m.indicator(on, amount == 0, active=0)
m.indicator(on, amount >= 1)
revenue = m.piecewise(amount, [0, 1, 3], [0, 4, 6])
active_amount = m.product(on, amount)
distance = m.absolute(amount - 2, exact=True)

m.maximize(revenue - 0.5*active_amount - distance - on)
r = m.solve(mip_rel_gap=0).require_optimal()
print(r[amount], r[on], r.objective)  # 約 2.0, 1.0, 3.0
```

indicator、絶対値、二値×連続積、SOS、区分線形は、LP/MILPへの再定式化として実装した。ネイティブの専用制約ハンドラではない。

自動Big-Mは変数の区間から導く。有限境界が不足した場合に適当な巨大値を入れることはせず、エラーにする。パラメータの変更後もMを再計算する。Mを直接指定する場合、その値が数学的に十分であることは利用者が確認する。

`absolute(exact=False)` や `abs_epigraph()` は $t\ge|x|$ の表現である。tを最小化するなど、等式に張り付く条件がなければ $t=|x|$ とは限らない。`exact=True` は有限境界と二値変数を使い、グラフそのものを表す。

多目的の辞書式求解、指定二値変数のパターン列挙、パラメータシナリオも同梱した。全連続解集合を列挙する機能とは区別している。

## 9. GLPK・HiGHSへの切替え

同じ線形モデルでバックエンドを変更できる。

```python
native = m.solve(solver="mylpk")
highs = m.solve(solver="highs")

# 先にプロジェクトディレクトリで次を実行する:
# python -m pip install '.[glpk]'
# glpk = m.solve(solver="glpk")
```

HiGHS経路はSciPyの[linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)・[milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)を使用する。GLPK経路は[swiglpk](https://pypi.org/project/swiglpk/)の直接APIで問題を生成し、ファイルを介さずに求解する。

外部ソルバーの設定と返却情報には差がある。独自の基底、コールバック、探索オプションがすべてのバックエンドで共通に使えるとはしていない。Pyomoを介する `pyomo:<solver>` の拡張経路も用意したが、外部ソルバー自身は別途必要である。

この成果物の環境ではswiglpk、highspy、Pyomoを取得できなかった。したがって、**GLPKとPyomoの連携、highspyを使ったLPファイル読込みは実機検証未実施**である。実装済みと検証済みを区別し、該当する条件付きテストをスキップとして記録している。

## 10. QP/QCQPは局所解として明示する

連続変数の二次目的・二次制約はモデルとして保持でき、明示的に `solver="scipy"` を指定するとSLSQPへ渡す。

```python
from myomo import Model

m = Model()
x = m.var("x", lb=-2, ub=2)
y = m.var("y", lb=-2, ub=2)
m += x*x + y*y <= 1
m.maximize(x+y)

r = m.solve(solver="scipy")
print(r.status)     # locally_optimal
print(r.feasible)   # True
print(r.success)    # False
print(r.objective)  # 約1.41421356237
```

SLSQPは[SciPy公式API](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)を利用する。解析的な勾配を渡すが、このアダプターでは凸性や大域最適性の証明をしていない。`success` を大域的な `optimal` だけに限定したため、局所解では偽になる。

MIQP、MINLP、SDP、SOCP、三次以上やsin/exp/logを含む一般NLPは、この版では未実装である。

## 11. 速度を実測する

測定はポータブルビルドで、ウォームアップ1回、5回の中央値である。BLASには1スレッドを指定した。HiGHS内部スレッド数は別途計測していない。プロセス起動やimportを除き、求解時の標準化・コピー・postsolve・検査を含む。

| ケース | 元の行数×変数数 | mylpk [ms] | HiGHS経路 [ms] | 目的値照合 |
|---|---:|---:|---:|---|
| dense_20x40 | 20×40 | 1.657 | 2.598 | 一致 |
| dense_80x160 | 80×160 | 3.415 | 6.901 | 一致 |
| sparse_120x600 | 120×600 | 24.875 | 11.551 | 一致 |
| transport_12x12 | 24×144 | 2.267 | 2.323 | 一致 |
| binary_knapsack_18 | 3×18 | 16.899 | 2.942 | 一致 |

小規模な密LPでは独自経路が短時間になった。一方、疎LPと二値整数問題ではHiGHS経路が短時間だった。これは少数の合成問題に対するAPI経路の比較であり、産業用ソルバー一般の性能ランキングではない。

目的係数を少しずつ変更する場合には、保持型のLPSessionで次の結果になった。

| 40行×100変数、初回＋20回の目的係数更新 | 合計時間の中央値 [ms] |
|---|---:|
| mylpk LPSession、初回を含む | 15.757 |
| mylpk、毎回新しく求解 | 44.762 |
| HiGHS経路、毎回新しく求解 | 74.444 |

21回分を合計し、LPSession側にも初回求解を含めた。HiGHS側はSciPyから毎回新規に呼んでいるため、HiGHSのネイティブpersistent APIとの比較ではない。標準化・コピー・LU再構成を省ける効果を含む結果として読む必要がある。

myomo内部の式構築については次のとおり。

| 変数項数 | 組込みsum [ms] | myomo.quicksum [ms] |
|---:|---:|---:|
| 1000 | 49.440 | 0.268 |
| 5000 | 1190.020 | 1.327 |

変数の作成・モデルコンパイル・求解は除き、一次式の合計だけを測っている。組込みsumが繰り返す辞書コピーをquicksumが避けるため、大きな差が生じた。**Cythonに変換しただけでこの倍率になるという実験ではない。Pyomoとの比較でもない。**

全ケースの生データと環境は `reports/benchmark.json`、再現コードは `benchmarks/run.py` に保存した。大規模標準問題集、native persistent HiGHS、Pyomoのモデル構築は未測定である。

## 12. 検証と配布構成

テストは **428件成功、21件スキップ**。スキップの内訳はswiglpk未導入によるGLPK向け20件、highspy未導入によるLP読込み1件である。これに加えて8本の実行例を実行した。

ランダム一般LP160ケースをHiGHSと比較し、二値MILP80ケースではHiGHSと全列挙による最適値を照合した。混合整数問題、退化、冗長等式、循環例、非有界方向、ウォームスタート、証明の改ざん検出、モデル更新後の解の無効化、疎バッファの不正入力、ファイル往復なども検査している。

ソースZIPからCython拡張を再ビルドし、別のインストール先に入れたwheelからも、428件成功・21件スキップと8本の実行例の成功を確認した。再現用スクリプトは `tools/validate_wheel.py` にある。ソース側のビルド済み `.so` を拾って成功したように見える状況を避けるため、実際のimport元も検査している。これは同じPython・第三者依存を使う検証であり、全OS・全Python版の検証ではない。

```text
mylpk/
  pyproject.toml
  setup.py
  src/mylpk/        独自ソルバー・解析・外部連携・入出力
  src/myomo/        数式・モデル・再定式化・ワークフロー
  tests/           回帰・照合・条件付き外部連携テスト
  examples/        実行例8本とCLI用JSON
  benchmarks/      再現可能な測定コード
  reports/         数値・テスト・実行例の記録
  docs/            API・数学・性能・制約事項
  tools/           ソースZIP作成・wheel独立検証
```

`.github`、ライセンスファイル、第三者ソルバー本体、生成済み拡張バイナリ、ビルドキャッシュは配布ZIPに含めていない。

## 結び

今回の構成で重点を置いたのは、数値コアのCython化、不要な再構築を避ける状態保持、数式を短く書くモデリングAPI、結果を検査する解析機能である。

速度については、型を付けることだけでなく、何を毎回計算し直しているかを分けて考える必要があった。小さな問題では呼出しや変換の負担が見え、反復更新では基底の再利用が効いた。一方、大きな疎基底や整数探索には、疎分解・高度なpresolve・カット・分枝戦略といった別の改善が必要になる。

初期版として、解ける問題、数値的な限界、実機未検証の経路を区別できる状態までを実装した。利用時には終了状態と元問題での制約違反を確認し、重要な問題では別ソルバーや証明チェックで照合する。
