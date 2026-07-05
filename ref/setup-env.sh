#!/usr/bin/env bash
# FlashInfer 开发环境一键配置脚本 (v0.6.14)
# 适用镜像: 缺少 CUDA 13.x cuTile 编译依赖，且不需要 cuTile 路径
#
# 用法:
#   bash scripts/setup-env.sh           # 全流程
#   bash scripts/setup-env.sh --skip-build  # 跳过最后的 pip install -e .
#
# 经验来源 (踩坑总结):
#   1. nvidia-cutlass-dsl 4.2.1 缺 OperandMajorMode -> import flashinfer 失败, 升级到 >=4.5.0
#   2. JIT 需要 cutlass/spdlog 头文件 -> git submodule update --init --recursive
#   3. 镜像无 CUDA 13.x cuTile 编译依赖, uv 下载会卡死; 且本机不用 cuTile -> patch build_backend.py
#   4. 环境内置 setuptools 75.8.0, pyproject 要求 >=77 (license 字段校验) -> 升级 setuptools
#   5. 残留 flashinfer-cubin 0.5.2 与 0.6.14 版本不匹配 -> 卸载
#   6. urllib3 1.26.4 vendor 的旧 six 注册了无 find_spec 的 importer, pytest 收集崩溃 -> 升级 urllib3>=2
#   7. FLASHINFER_JIT_VERBOSE=1 等价于 debug 模式 (-g -O0 --device-debug), 编译极慢 -> 关闭
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --skip-build) SKIP_BUILD=1 ;;
    *) echo "unknown arg: $arg" >&2 ;;
  esac
done

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 0. 前置: 关闭 JIT verbose/debug, 避免编译极慢或中断 (经验 7)
# ---------------------------------------------------------------------------
log "导出 JIT 环境变量 (debug=0, verbose=0)"
export FLASHINFER_JIT_DEBUG=0
export FLASHINFER_JIT_VERBOSE=0

# ---------------------------------------------------------------------------
# 1. 升级 nvidia-cutlass-dsl (经验 1): 4.2.1 缺 OperandMajorMode
# ---------------------------------------------------------------------------
log "升级 nvidia-cutlass-dsl >= 4.5.0"
pip install -U "nvidia-cutlass-dsl>=4.5.0"

# ---------------------------------------------------------------------------
# 2. 初始化子模块 (经验 2): JIT 需要 cutlass / spdlog / cccl 头文件
# ---------------------------------------------------------------------------
log "初始化 git 子模块 (cutlass / spdlog / cccl)"
git submodule update --init --recursive

# ---------------------------------------------------------------------------
# 3. patch build_backend.py (经验 3): 跳过 cuTile 编译依赖安装
# ---------------------------------------------------------------------------
log "patch build_backend.py — _install_cuda_tile_compile_deps early-return"
BB="$REPO_ROOT/build_backend.py"
if ! grep -q "PATCHED_SKIP_CUDA_TILE" "$BB"; then
  python - "$BB" <<'PY'
import sys, ast
p = sys.argv[1]
s = open(p).read()
needle = "def _install_cuda_tile_compile_deps() -> None:\n"
assert needle in s, "函数签名未找到, 检查 build_backend.py 是否变更"
# 在函数体最前面插入 early-return, 原始 docstring/逻辑作为 dead code 保留 (语法合法).
s = s.replace(
    needle,
    "def _install_cuda_tile_compile_deps() -> None:\n"
    "    # PATCHED_SKIP_CUDA_TILE: 镜像无 CUDA 13.x cuTile 编译依赖, uv 下载卡死;\n"
    "    # 本环境不使用 cuTile 路径, 直接 early-return 跳过.\n"
    "    print('[BUILD] (patched) skip cuTile compile-deps install', flush=True)\n"
    "    return\n",
)
ast.parse(s)  # 确保替换后语法合法
open(p, "w").write(s)
print("patched OK")
PY
else
  echo "  已 patch, 跳过"
fi

# ---------------------------------------------------------------------------
# 4. 升级 setuptools>=77 (经验 4): pyproject license 字段校验要求
# ---------------------------------------------------------------------------
log "升级 setuptools >= 77"
pip install -U "setuptools>=77"

# ---------------------------------------------------------------------------
# 5. 卸载残留 flashinfer-cubin (经验 5): 避免 0.5.2 与 0.6.14 版本不匹配
# ---------------------------------------------------------------------------
log "卸载残留 flashinfer-cubin (如有)"
pip uninstall -y flashinfer-cubin 2>/dev/null || true

# ---------------------------------------------------------------------------
# 6. 升级 urllib3>=2 (经验 6): 修复 pytest 收集崩溃 (旧 six importer)
# ---------------------------------------------------------------------------
log "升级 urllib3 >= 2"
pip install -U "urllib3>=2"

# ---------------------------------------------------------------------------
# 7. 安装 flashinfer (editable, no-build-isolation)
# ---------------------------------------------------------------------------
if [[ "$SKIP_BUILD" -eq 1 ]]; then
  log "跳过 pip install -e . (--skip-build)"
else
  log "pip install --no-build-isolation -e . -v"
  pip install --no-build-isolation -e . -v
fi

# ---------------------------------------------------------------------------
# 8. 冒烟验证
# ---------------------------------------------------------------------------
log "冒烟验证: import flashinfer"
python - <<'PY'
import flashinfer
print("flashinfer version:", flashinfer.__version__)
PY

log "完成. 推荐日常环境变量:"
cat <<'EOF'
  export FLASHINFER_JIT_DEBUG=0      # 关闭 -g -O0 --device-debug, 编译更快
  export FLASHINFER_JIT_VERBOSE=0    # 关闭 verbose (避免等价 debug 模式)
  # 可选: 限制并行编译, 防止内存爆
  # export MAX_JOBS=4
  # export FLASHINFER_NVCC_THREADS=4
  # 可选: 指定架构, 加快首轮编译
  # export FLASHINFER_CUDA_ARCH_LIST="8.0 9.0a"
EOF