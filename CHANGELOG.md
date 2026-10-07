# Changelog

## 0.1.0 (2026-10-07)

独自Cython改訂単体法、Python分枝限定法、myomoの記号モデリングと線形再定式化を追加。

元座標KKT検査、双対問題、標準形感度、IIS、有理数によるLP最適性証明チェックを追加。

LU・eta保持型LPSession、明示的なHiGHS/GLPK/Pyomoアダプター、局所QP/QCQPのSLSQP経路、JSON/MPS/LP入出力、CLIを追加。

Python 3.13/Cython 3.2のビルドと、ソースからのwheel再ビルド検証、テスト、実行例、全ケース掲載のベンチマークを同梱。

外部依存が存在しないバックエンドは未検証として明記。汎用的な最速・全機能対応を宣言しない初期版として公開準備。
