#!/usr/bin/env python3
"""
GPU環境テストスクリプト
WSL2 + RTX 3080 + Driver 566.14環境でのPyTorch GPU動作確認
"""

import sys

print("=" * 60)
print("GPU環境診断ツール")
print("=" * 60)
print()

# 1. PyTorchのインポート確認
print("1. PyTorchインポート確認...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"   ✗ PyTorchのインポートに失敗: {e}")
    sys.exit(1)

# 2. CUDA利用可能性の確認
print("\n2. CUDA利用可能性確認...")
if torch.cuda.is_available():
    print(f"   ✓ CUDAが利用可能")
    print(f"   ✓ CUDAバージョン: {torch.version.cuda}")
    print(f"   ✓ cuDNNバージョン: {torch.backends.cudnn.version()}")
else:
    print(f"   ✗ CUDAが利用できません")
    print(f"   CUDA built: {torch.version.cuda}")
    sys.exit(1)

# 3. GPU情報の表示
print("\n3. GPU情報...")
gpu_count = torch.cuda.device_count()
print(f"   GPU数: {gpu_count}")
for i in range(gpu_count):
    print(f"\n   GPU {i}:")
    print(f"     名前: {torch.cuda.get_device_name(i)}")
    print(f"     計算能力: {torch.cuda.get_device_capability(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"     総メモリ: {props.total_memory / 1024**3:.2f} GB")
    print(f"     マルチプロセッサ数: {props.multi_processor_count}")

# 4. 簡単な演算テスト
print("\n4. GPU演算テスト...")
try:
    # テンソルをGPUに配置
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    
    # 行列乗算
    z = torch.matmul(x, y)
    
    print(f"   ✓ GPU演算成功")
    print(f"   ✓ テンソル形状: {z.shape}")
    print(f"   ✓ テンソルデバイス: {z.device}")
    
    # メモリ使用状況
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    print(f"   ✓ 割り当てメモリ: {allocated:.2f} MB")
    print(f"   ✓ 予約メモリ: {reserved:.2f} MB")
    
except Exception as e:
    print(f"   ✗ GPU演算に失敗: {e}")
    sys.exit(1)

# 5. PyTorch設定確認
print("\n5. PyTorch設定...")
print(f"   cuDNN有効: {torch.backends.cudnn.enabled}")
print(f"   cuDNNベンチマーク: {torch.backends.cudnn.benchmark}")

print("\n" + "=" * 60)
print("✓ すべてのテストに合格しました！")
print("PyTorchはGPUを正しく使用できます。")
print("=" * 60)
