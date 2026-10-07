# ビルドと再現手順

## 検証した構成

CPython 3.13.5、Cython 3.2.4、NumPy 2.3.5、SciPy 1.17.0、GCC 14.2.0、Linux x86-64、LP64版LAPACKで検証しました。SciPyのCython関数カプセルからLAPACKを呼びます。別のGLPK実行ファイルは独自ソルバーのビルドに不要です。

メタデータは `Python >=3.10`、`numpy>=1.26`、`scipy>=1.11,<1.18` としています。これは依存解決の許容範囲です。各OS・Python版・依存版の全組合せに対する試験済み宣言ではありません。Windows、macOS、ARM、PyPy、free-threaded CPythonでは未検証です。

数値コアは32ビットCSC/LAPACK整数を使用します。SciPy 1.18から文書化された新しい `blas_int` / ILP64対応はこの版の対象外です。上限を削って依存だけ更新せず、整数型、ピボット配列、Cython関数シグネチャを変更し、再ビルド・検証してください。

## 通常のインストール

ZIPを展開し、`pyproject.toml` があるディレクトリへ移動します。

```bash
python3.13 -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[dev]'
python -m pytest -q
```

Cコンパイラと対応するPython開発ヘッダが必要です。Debian系OSのシステムPythonなら、通常は次のパッケージを使用します。独自に導入したPythonには、そのPythonに対応したヘッダが必要です。

```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev python3-venv
```

ネットワークに接続できる環境では、pipのビルド分離が `pyproject.toml` のビルド依存を準備します。本成果物の検証環境では外部パッケージ取得ができなかったため、導入済み依存を使う `--no-build-isolation --no-deps` のビルド経路を検証しています。

## 開発ビルド

```bash
python -m pip install 'setuptools>=74' wheel 'Cython>=3.1,<4' \
  'numpy>=1.26' 'scipy>=1.11,<1.18' pytest threadpoolctl
python setup.py build_ext --inplace
PYTHONPATH=src python -m pytest -q
```

`setup.py build_ext --inplace` は開発・診断用です。通常の配布物のインストールにはpipを使います。`requirements-dev.txt` は今回の主要数値依存とテスト依存を固定した再現用ファイルです。対応するwheelのない環境で同じ構成が導入できるという保証はありません。

## CPUに合わせた最適化

既定値はGCC/Clangで `-O3 -fno-math-errno`、MSVCで `/O2` です。GCC/Clang環境では次の指定でホストCPU向け命令を有効にできます。

```bash
MYLPK_NATIVE=1 python -m pip install --no-build-isolation --no-deps \
  --force-reinstall --no-cache-dir .
```

`-march=native -mtune=native` を加えます。生成したwheelは、異なるCPUでは実行できない可能性があります。このオプションで必ず高速になるとは限らず、同梱の性能測定は**既定のポータブルビルド**です。

`-ffast-math` は使用しません。無限大・NaN・実行可能性・最適性判定を変更して得た時間短縮は、数理最適化の正しい高速化として評価しない方針です。

Cython注釈HTMLを生成するには次を使います。

```bash
MYLPK_ANNOTATE=1 python setup.py build_ext --inplace --force
```

## ソースZIPからwheelを独立検証する

```bash
python tools/validate_wheel.py
```

このスクリプトは一時ディレクトリにソースZIPを生成・展開し、pipでwheelをビルドし、別ディレクトリへインストールして、インストール済みパッケージからテスト・実行例を実行します。ソース側の `.so` を流用しません。第三者の依存ライブラリとPythonインタープリタは同じ環境を使用します。完全に別のOSやコンテナを検証するものではありません。

## よくある問題

`ModuleNotFoundError: mylpk._core` はCython拡張が未ビルド、または別のPython用にビルドされた場合に発生します。対象のPythonでpipインストールするか、開発ビルドを実行してください。

`scipy.linalg.cython_lapack` に関する型・カプセルのエラーがある場合は、SciPyの版・LAPACK整数ABI・ビルド時と実行時の環境を確認し、対応範囲内で再ビルドしてください。既存の `.so` を別環境へコピーする方法は使用しません。

`MemoryError` が基底ワークスペースを指している場合は、密LUの必要メモリが上限を超えています。`max_basis_mb` は安全装置であり、値を上げても疎基底分解には変わりません。大規模疎LPでは `solver="highs"` を検討してください。

GLPKのImportErrorでは `python -m pip install '.[glpk]'`、LPファイル読込みでは `python -m pip install '.[interop]'` を使用します。Pyomoブリッジの外部ソルバーは別途インストールする必要があります。
