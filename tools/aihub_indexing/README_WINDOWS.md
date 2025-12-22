# Windows 빠른 설치

## 1) (권장) Python 3.11로 한 방 설치
```powershell
powershell -ExecutionPolicy Bypass -File tools/aihub_indexing/windows_setup.ps1
```

## 2) 직접 설치(현재 venv 사용)
```powershell
python -m pip install -U pip
python -m pip install -r tools/aihub_indexing/requirements_indexing.txt
```

- `hnswlib`가 없으면 `2_embed_index.py`는 자동으로 `*_flat.npy` 인덱스를 만듭니다.
- `hnswlib`가 꼭 필요하면(Windows는 Python 3.11 venv 권장):
```powershell
python -m pip install -r tools/aihub_indexing/requirements_hnsw_optional.txt
```
