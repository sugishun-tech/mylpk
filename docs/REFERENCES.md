# 参考資料

確認日: 2026-10-07。以下はAPI・ファイル形式・設計上の判断の参照先です。リンク先の最新版と、実際のビルド検証に使用した版は区別しています。測定結果と実装範囲の根拠は、このプロジェクトのソースと `reports/` です。

- [Cython: Typed Memoryviews](https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html)
- [SciPy: LAPACK functions for Cython](https://docs.scipy.org/doc/scipy/reference/linalg.cython_lapack.html)。新しい文書のILP64/blas_int対応は本版の対象外で、依存をSciPy 1.18未満に制限しています。
- [SciPy: linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)
- [SciPy: milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
- [SciPy: SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
- [HiGHS公式サイト](https://highs.dev/)
- [swiglpk: パッケージ説明とGLPK API使用例](https://pypi.org/project/swiglpk/)
- [Pyomo: Simple Examples](https://pyomo.readthedocs.io/en/stable/getting_started/pyomo_overview/simple_examples.html)
- [Pyomo: Expressions](https://pyomo.readthedocs.io/en/stable/explanation/modeling/math_programming/expressions.html)
- [Gurobi: Model File Formats](https://docs.gurobi.com/projects/optimizer/en/current/reference/fileformats/modelformats.html)。MPS integer markerの既定境界などの形式参照であり、Gurobiを実行検証したという意味ではありません。
- [lp_solve: MPS file format](https://lpsolve.sourceforge.net/5.5/mps-format.htm)。MPS方言間の境界・目的定数解釈の差を確認しています。

本プロジェクトはこれらのライブラリ・プロジェクトとは独立しています。第三者のソルバー本体や第三者のソースコードは配布ZIPに取り込んでいません。
