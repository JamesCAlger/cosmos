@echo off
echo Opening Week 3 Evaluation Reports...
echo.

echo Reports are located in:
echo %~dp0experiments\week3_msmarco\
echo.

echo Opening HTML report in browser...
start "" "%~dp0experiments\week3_msmarco\week3_msmarco_main.html"

echo Opening reports folder...
explorer "%~dp0experiments\week3_msmarco\"

echo.
echo Available reports:
echo - week3_msmarco_main.html (Visual report - best for viewing)
echo - week3_msmarco_main.json (Complete data)
echo - week3_msmarco_main.markdown (Summary)
echo - EVALUATION_SUMMARY.md (Detailed summary)
echo.
pause