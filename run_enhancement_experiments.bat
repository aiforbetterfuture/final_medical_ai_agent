@echo off
REM Agentic RAG 고도화 실험 자동 실행 스크립트 (Windows)
REM 사용법: run_enhancement_experiments.bat

setlocal enabledelayedexpansion

echo ==========================================
echo Agentic RAG 고도화 실험 자동 실행
echo ==========================================
echo.

REM ============================================================
REM Phase 1: RAG 변형 비교 (피드백 반영)
REM ============================================================

echo [Phase 1] RAG 변형 비교 실험
echo.

REM 환자 시나리오 3개 실행
for %%P in (P001 P002 P003) do (
    echo [실행] 환자 시나리오: %%P
    
    python experiments\run_rag_variants_comparison.py ^
        --patient-id %%P ^
        --turns 5 ^
        --variants basic_rag modular_rag corrective_rag
    
    if errorlevel 1 (
        echo [오류] %%P 실험 실패
        pause
        exit /b 1
    )
    
    echo.
)

echo [완료] Phase 1: RAG 변형 비교 로그 생성
echo.

REM ============================================================
REM Phase 2: RAGAS 평가
REM ============================================================

echo [Phase 2] RAGAS 평가 (LLM as a Judge)
echo.

set COMPARISON_DIR=runs\rag_variants_comparison

if not exist "%COMPARISON_DIR%" (
    echo [경고] 비교 결과 디렉토리가 없습니다: %COMPARISON_DIR%
    pause
    exit /b 1
)

REM 각 환자별 최신 파일 평가
for %%P in (P001 P002 P003) do (
    REM 최신 파일 찾기 (PowerShell 사용)
    for /f "delims=" %%F in ('powershell -Command "Get-ChildItem '%COMPARISON_DIR%\comparison_%%P_*.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do (
        set LATEST_FILE=%%F
    )
    
    if defined LATEST_FILE (
        echo [평가] %%P: !LATEST_FILE!
        
        python experiments\evaluate_rag_variants.py "!LATEST_FILE!"
        
        if errorlevel 1 (
            echo [오류] %%P 평가 실패
            pause
            exit /b 1
        )
        
        echo.
    ) else (
        echo [경고] %%P 비교 결과 파일이 없습니다
    )
)

echo [완료] Phase 2: RAGAS 평가 결과 생성
echo.

REM ============================================================
REM Phase 3: 결과 요약
REM ============================================================

echo [Phase 3] 결과 요약
echo.

set RAGAS_DIR=%COMPARISON_DIR%\ragas_evaluation

if exist "%RAGAS_DIR%" (
    echo [요약] RAGAS 평가 결과:
    echo.
    
    for %%F in (%RAGAS_DIR%\ragas_summary_*.csv) do (
        echo 파일: %%~nxF
        type "%%F"
        echo.
    )
    
    echo [완료] Phase 3: 결과 요약 출력
) else (
    echo [경고] RAGAS 평가 디렉토리가 없습니다
)

echo.

REM ============================================================
REM 완료 메시지
REM ============================================================

echo ==========================================
echo [완료] 모든 실험 완료!
echo ==========================================
echo.
echo 결과 위치:
echo   - 비교 로그: %COMPARISON_DIR%\comparison_*.json
echo   - RAGAS 평가: %RAGAS_DIR%\ragas_*.json
echo   - CSV 요약: %RAGAS_DIR%\ragas_summary_*.csv
echo.
echo 다음 단계:
echo   1. CSV 파일을 엑셀/구글 시트로 열어 테이블 작성
echo   2. JSON 파일에서 통계적 유의성 확인 (p-value, Cohen's d)
echo   3. 논문/보고서 작성
echo.

pause

