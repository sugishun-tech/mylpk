# APIリファレンス

## myomo.Model

```python
from myomo import Model, quicksum, dot, between

m = Model("example")
x = m.var("x", lb=0, ub=10, kind="continuous")
y = m.var("free", lb=None)
b = m.var("enabled", kind="binary")
x.fix(3)

v = m.vars("v", 10, lb=0, ub=5)
ship = m.vars("ship", ["A", "B"], ["X", "Y"], lb=0)
capacity = m.param("capacity", 10)

row = m.add(2*x + v.sum() <= capacity, name="capacity_row")
m += between(0, y, 4)
m.add(ship.sum(i, "*") <= 5 for i in ["A", "B"])
m.minimize(x + quicksum(v))
```

`kind` は `continuous` / `integer` / `binary` / `semicontinuous` / `semiinteger`。半変数には非負下限と有限上限が必要です。`vars()` は整数長、反復可能な添字集合、複数の集合による直積を受け取ります。`lb` / `ub` はスカラー、辞書、対応する一次元配列、添字を受け取る関数にもできます。

IndexedVarsを反復すると**キーでなく変数**が得られます。キーは `.keys()`、対応は `.items()` を使います。`.sum("A", "*")` の `"*"` はその添字位置の全要素です。`.dot()` はキー対応の辞書または順序対応の係数列を受け取ります。

`Parameter.value` は変更可能です。式の係数、定数、変数境界に使えます。値の更新、変数境界・種別の変更、制約上下限の変更、公開モデル操作は解キャッシュを無効にします。パラメータで決定変数を割ることはできますが、決定変数そのものによる除算は未対応です。

大きな和は `quicksum()` で一度だけ係数を集計してください。Pythonの `sum()` は式のコピーを繰り返します。式の辞書や内部の配列は直接変更しません。

## 行列の一括追加

```python
from scipy import sparse
m = Model()
x = m.vars("x", 3)
rows = m.add_matrix(sparse.csr_matrix([[1, 2, 0], [0, 1, 1]]), x,
                    lb=[-float("inf"), 2], ub=[8, 2], name="bulk")
m.minimize(x.dot([1, 2, 3]))
r = m.solve().require_optimal()
print(rows.dual)
```

列の順序は渡した変数列と対応します。MatrixBlockの行列は作成時のスナップショットとして扱います。変更には新しいモデル/ブロックを作るか、許可されたパラメータを使用してください。

## 求解と結果

```python
r = m.solve(solver="mylpk", time_limit=30, mip_rel_gap=0)
if r.success:
    print(r.objective, r[x])
elif r.feasible:
    print(r.status, r.objective, r.bound, r.gap)
else:
    print(r.status, r.message)
```

`r[var]`、`r[indexed_vars]` は返却スナップショットの値です。`var.value` は**現在のモデル版**に対する実行可能解がある場合だけ取得できます。`constraint.dual` は現在の連続LP解がある場合に取得できます。MILPで一般の双対値を返すことはありません。

主なフィールドは `status, x, objective, fun, solver, message, iterations, nodes, bound, gap, elapsed, activity, max_violation, dual, reduced_costs, row_lower_dual, row_upper_dual, lower_dual, upper_dual, basis, ray, certificate, raw` です。`fun` は `objective` の別名です。`max_violation` は絶対量で、`raw["scaled_primal_violation"]` はスケールを考慮した検査値です。

最小化LPでは下限側の双対値は非負、上限側は非正です。最大化では符号が反転します。等式乗数は自由です。これらは元の目的関数の向き・係数スケールへ戻した値で、常に一意とは限りません。

| 状態 | 意味 |
|---|---|
| `optimal` | 許容誤差内の最適終了。`success=True` |
| `infeasible` | 実行不可能と判定 |
| `unbounded` | 目的が非有界。独自LPでは改善方向も返却 |
| `relaxation_unbounded` | MILPのLP緩和が非有界。整数問題自体の結論ではない |
| `infeasible_or_unbounded` | 外部ソルバーの判定が両者を区別していない |
| `iteration_limit`, `time_limit`, `node_limit`, `limit` | 制限に達した。実行可能解がある場合もある |
| `gap_limit` | 独自MILP/外部ソルバーで指定相対ギャップに到達。独自ソルバーでは `success=False` |
| `user_stop` | MILPコールバックが停止を要求 |
| `numerical_error`, `solver_error`, `invalid_basis` | 数値検査失敗、外部エラーなど |
| `locally_optimal` | 連続QP/QCQPのSLSQPによる局所終了。`success=False` |
| `local_solver_failure` | 局所ソルバーが成功終了しなかった |

`require_optimal()` は `optimal` 以外でRuntimeErrorを発生させます。`to_dict()` は基本的な結果を辞書へ変換しますが、`raw` の内部ワークスペースや証明をすべて保存する機能ではありません。

## mylpkの行列API

```python
import numpy as np
from mylpk import LinearProblem, solve, linprog, milp

p = LinearProblem(
    c=[-3, -2], A=[[1, 1], [1, 0], [0, 1]],
    row_upper=[4, 2, 3], lower=0, upper=np.inf,
    integrality=0, sense="min", offset=0,
)
r = solve(p).require_optimal()
q = p.replace(c=[-2, -4])
r2 = solve(q, basis=r.basis).require_optimal()

r3 = linprog([-3, -2], A_ub=[[1, 1]], b_ub=[4],
             bounds=[(0, 2), (0, 3)])
r4 = milp([-3, -2], A=[[2, 1], [1, 2]], row_upper=[14, 14],
          lower=0, integrality=1)
```

`LinearProblem` の行列はNumPy配列・SciPy疎行列を受け取り、検証・コピー・重複統合後にCSCで保持します。`integrality` は0=連続、1=整数、2=半連続、3=半整数です。`.replace()` は新しい問題を作り、`.relax()` は整数性を緩和します。保存済み配列の直接変更はしないでください。

`linprog` はSciPyに似た引数名を採用していますが、SciPyの全引数・全結果項目に互換ではありません。`milp` も本プロジェクトの行上下限APIで、SciPy `milp` のドロップイン互換ではありません。

## 独自ソルバーのOptions

`solve(p, options=Options(...))`、`options={...}`、直接のキーワード引数のいずれでも指定できます。未知のオプションを黙って受け取る仕組みはありません。

| 名前 | 既定値 | 対象・内容 |
|---|---:|---|
| `max_iter` | 100000 | LPの総ピボット上限。MILPでは各LP呼出しの上限 |
| `time_limit` | 無限大 | 求解の時間制限。チャンク・ノード境界で確認 |
| `feasibility_tol` | 1e-7 | 主実行可能性などの検査閾値 |
| `dual_tol` | 1e-9 | LPの双対実行可能性 |
| `integrality_tol` | 1e-7 | 整数性・分岐候補の判定 |
| `pivot_tol` | 1e-12 | 小さなピボットの判定 |
| `refactor_interval` | 48 | LU再分解間のeta最大本数。1〜1024 |
| `pivot_rule` | `dantzig` | `dantzig` / `bland` |
| `scaling` | True | 行・目的係数スケーリング |
| `presolve` | True | 定数行除去。固定変数の標準化は常に実施 |
| `max_basis_mb` | 512 | 密基底ワークスペース推定量の上限。プロセス全体の上限ではない |
| `node_limit` | 100000 | MILPの処理ノード数上限 |
| `mip_rel_gap` | 1e-6 | MILP相対ギャップ |
| `mip_abs_gap` | 1e-8 | MILP絶対目的差・枝刈り閾値 |
| `cuts` | True | 根ノード二値カバーカット |
| `heuristic` | True | 整数固定LPによる丸めヒューリスティック |

独自MILPの `x0=` は元変数空間の実行可能な整数暫定解です。`callback(event)` がFalseを返すと停止します。eventには `nodes, incumbent, bound, elapsed, cuts` が入ります。モデルを変更するコールバック、任意カット追加には対応しません。独自LPは `basis=` を受け取り、`x0` / `callback` は受け取りません。

## LPSession・独立並列求解

```python
from mylpk import LPSession, solve_many

session = LPSession(p)
r = session.resolve(c=[-2, -4])
r = session.resolve(row_upper=[3, 2, 3])
results = solve_many([p, q], workers=2)
```

`resolve` の変更可能項目は `c, row_lower, row_upper, offset, sense`。RHSの有限/無限、等式か否かの構造は固定です。行列・変数境界の変更は新しいLPSessionを作ってください。`raw["persistent_workspace"]` と `raw["cold_restart"]` で再利用の有無が分かります。

`m.solve(warm_start=True)` は前回の基底を渡す便利機能です。毎回モデルをコンパイルし、LU保持はしません。継続的な係数・RHS更新でLUまで再利用するには `LPSession` を使います。

## 線形再定式化

```python
m.indicator(b, x <= 2, active=1)       # 自動Big-M。有限境界が必要
m.indicator(b, x <= 2, M=10)          # Mの妥当性は利用者が担保
u = m.absolute(x - 3, exact=True)      # 二値変数を使う厳密グラフ
v = m.abs_epigraph(x - 3)              # v >= |x - 3|
y = m.product(b, x)                   # 二値×有界アフィン式
f = m.piecewise(x, [0, 2, 5], [0, 3, 4])
t = m.max_epigraph([x, 2*x - 1])
s = m.min_hypograph([x, 2*x - 1])
penalty = m.soft(x >= 4, penalty=100)  # 目的に追加する式を返す
```

`soft` は目的を自動変更しません。最小化なら正のペナルティとして足し、最大化なら引くなど、目的の方向を明示してください。`absolute(exact=False)` はepigraphです。

`logical_and(vars)` / `logical_or(vars)` / `logical_xor(a,b)` は二値出力を返します。`all_different(vars)` は有限境界を持つ整数変数を対ごとに分けるO(n²)規模の定式化です。`sos1(vars)` / `sos2(vars)` は並び順に対する二値再定式化で、SOS2では隣接する最大二要素だけが非ゼロになれます。

## ワークフロー

```python
r = m.solve_lexicographic([(objective1, "min"), (objective2, "max")],
                          atol=1e-8, rtol=0)
solutions = m.enumerate_solutions(10, variables=binary_variables)
results = m.solve_scenarios({"case_a": {"capacity": 8}, "case_b": {"capacity": 12}})
```

上記はそれぞれ独立した使用例で、`objective1` 等は対象モデルの式です。辞書式最適化では、前段目的値を `atol + rtol*abs(value)` の範囲で保ちます。`r.objective` は最後に解いた目的の値です。前段の値は `r.raw["lexicographic_values"]` にあります。

列挙は指定二値パターンへのno-good cutを一つずつ加え、現在の目的で最適化を繰り返します。すべての解が元問題の同率最適解とは限りません。補助二値変数を含めるかは `variables` で選んでください。シナリオごとに元のパラメータを基準とし、終了後に元の値を復元します。結果は `r[var]` で読むスナップショットで、元モデルの `.value` は再求解するまで利用できません。

## 数学的解析

`dual_problem(p)` はLP双対を新しいLinearProblemとして返します。`kkt_report(p,r)` は元座標の主制約違反、双対符号違反、停留性、主双対目的差を返します。

`basis_sensitivity(p,r)` は `rhs_delta` / `cost_delta` / `row_provenance` / `row_factor` / `objective_scale` を返します。差分区間は標準形座標であり、元係数への自動変換はしていません。密解析のため既定で実列数2000までです。

`find_iis(p, include_bounds=True, relax_integrality=True, max_checks=1000)` は `members`、`irreducible`、実行した `checks` を返します。制限や未確定判定があれば `partial` になり、完了したふりをしません。

`make_exact_certificate(p,r,max_denominator=10**9)` は有理数再構成と検査を行い、`certified`、`failures`、`certificate`、有理数文字列の主双対目的値を返します。`verify_exact_certificate(p, certificate)` は保存された証明を再検査できます。

## 外部バックエンド

| 指定 | 経路 | 主に転送する共通設定 | この成果物での検証 |
|---|---|---|---|
| `highs` のLP | SciPy linprog | presolve、時間、反復上限、主・双対許容誤差 | 実行済み |
| `highs-ds` / `highs-ipm` | SciPy linprog | 同上。連続LPのみ | インターフェース実装。既定highsを主に照合 |
| `highs` のMILP | SciPy milp | presolve、時間、ノード上限、相対ギャップ | 実行済み |
| `glpk` | swiglpk直接API | presolve、時間、LP反復、LP許容誤差、MIP整数性・相対ギャップ | 未実施 |
| `pyomo:<name>` | Pyomo SolverFactory | `backend_options` のみ | 未実施 |

独自Optionsのすべてが外部ソルバーへ対応するわけではありません。外部ソルバーの独自設定は `backend_options={...}` で渡します。SciPyアダプターは公開オプションの許可リストで検査します。独自 `cuts` / `heuristic` / `refactor_interval` 等は独自ソルバー専用です。

GLPKで `node_limit` や `mip_abs_gap` を既定値以外にするとエラーになります。既定値のこれらの制限もGLPKへは転送されません。GLPKの列挙値は有効値を検査してからC APIへ渡します。Pyomo経路では共通Optionsを既定値から変更するとエラーです。必要な時間制限等は当該ソルバーの名前で `backend_options` に指定します。GLPKの同時呼出しはロックで直列化します。

外部アダプターは `basis` / `x0` / `callback` を公開していません。未対応指定を受けたときはエラーで止め、無視しません。バックエンドは失敗時に勝手に別のソルバーへ切り替えません。

`register_solver(name, function)` で `(problem, options, backend_options) -> SolveResult` を満たすソルバーを登録できます。`available_solvers()` は組込み・登録済み経路の導入状態を返します。Pyomo配下の全外部ソルバーの検出一覧ではありません。

## 入出力・CLI

```python
from mylpk import read_problem
p.write("model.json")
p.write("model.mps")
p.write("model.lp")
q = read_problem("model.json")
```

JSONは `mylpk.linear.v1` のスナップショットで、元の名前を保持します。MPS/LP書出しは安全な `x0, x1, ...` / `r0, r1, ...` の名前へ正規化します。QP/QCQPの書出しは未対応です。LP読込みはhighspyを使い、二次目的を検出した場合は線形問題として黙って取り込まず拒否します。

MPSは自由形式の線形部分を対象とし、ROWS/COLUMNS/RHS/RANGES/BOUNDS、OBJSENSE/OBJNAME、integer markerを扱います。セミ変数の書出しは二値変数へ展開します。目的定数は `RHS[OBJ] = -offset` の規約を採用します。

integer marker内の変数にBOUNDS記録が全くないときは、既定で0/1境界を採用します。別規約には `mylpk.io.read_mps(path, integer_default="unbounded")` を指定します。一つでも明示的な境界記録がある場合、未指定の上限は通常の無限大になります。負の上限だけがあり下限が明示されない曖昧な記録は拒否します。MI/LOで下限を明示してください。複数のRHS/RANGES/BOUNDS集合では、既定で最初の集合を使い、同関数の `rhs_name` / `ranges_name` / `bounds_name` で選択できます。

MPSのSOS・二次・indicator等の未対応セクションは拒否します。出力するOBJSENSE拡張を解釈しないソルバーもあり、すべてのMPS読込み実装との互換性を保証しません。myomo→GLPKの直接経路はMPSファイルを介しません。

```bash
mylpk solvers
mylpk inspect examples/production.json
mylpk solve examples/production.json --solver mylpk --time-limit 30 --output result.json
mylpk convert examples/production.json production.mps
```

CLIはPythonモデルファイルを実行しません。終了コードは0=最適、2=実行不可能/非有界、3=制限等のその他終了、4=入力・実行エラーです。
