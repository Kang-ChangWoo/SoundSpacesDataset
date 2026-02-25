#!/bin/bash
# Compare two dataset folders to verify they have matching contents.
# Checks: scene count, subfolder structure, file counts, and total sizes.
#
# Usage:
#   bash check_both_folder.sh /path/to/folder_A /path/to/folder_B

if [ $# -ne 2 ]; then
    echo "Usage: bash check_both_folder.sh <folder_A> <folder_B>"
    exit 1
fi

DIR_A="$1"
DIR_B="$2"
ERRORS=0
WARNINGS=0

echo "=========================================="
echo "Comparing two dataset folders"
echo "  A: $DIR_A"
echo "  B: $DIR_B"
echo "=========================================="
echo ""

# --- 1. Scene count ---
echo "[1/4] Checking scene counts..."
SCENES_A=$(ls -1d "${DIR_A}"/*/ 2>/dev/null | xargs -I{} basename {} | sort)
SCENES_B=$(ls -1d "${DIR_B}"/*/ 2>/dev/null | xargs -I{} basename {} | sort)
COUNT_A=$(echo "$SCENES_A" | grep -c .)
COUNT_B=$(echo "$SCENES_B" | grep -c .)
echo "  A: $COUNT_A scenes"
echo "  B: $COUNT_B scenes"

ONLY_A=$(comm -23 <(echo "$SCENES_A") <(echo "$SCENES_B"))
ONLY_B=$(comm -13 <(echo "$SCENES_A") <(echo "$SCENES_B"))

if [ -n "$ONLY_A" ]; then
    echo "  ERROR: Only in A:"
    echo "$ONLY_A" | sed 's/^/    - /'
    ERRORS=$((ERRORS + 1))
fi
if [ -n "$ONLY_B" ]; then
    echo "  ERROR: Only in B:"
    echo "$ONLY_B" | sed 's/^/    - /'
    ERRORS=$((ERRORS + 1))
fi
[ -z "$ONLY_A" ] && [ -z "$ONLY_B" ] && echo "  OK"

COMMON_SCENES=$(comm -12 <(echo "$SCENES_A") <(echo "$SCENES_B"))
echo ""

# --- 2. Subfolder structure ---
echo "[2/4] Checking subfolder structure..."
SUB_ERR=0
for scene in $COMMON_SCENES; do
    SUBS_A=$(ls -1 "${DIR_A}/${scene}/" 2>/dev/null | sort)
    SUBS_B=$(ls -1 "${DIR_B}/${scene}/" 2>/dev/null | sort)
    DIFF_AB=$(comm -23 <(echo "$SUBS_A") <(echo "$SUBS_B"))
    DIFF_BA=$(comm -13 <(echo "$SUBS_A") <(echo "$SUBS_B"))
    if [ -n "$DIFF_AB" ] || [ -n "$DIFF_BA" ]; then
        echo "  MISMATCH [$scene]"
        [ -n "$DIFF_AB" ] && echo "$DIFF_AB" | sed 's/^/    only in A: /'
        [ -n "$DIFF_BA" ] && echo "$DIFF_BA" | sed 's/^/    only in B: /'
        SUB_ERR=$((SUB_ERR + 1))
    fi
done
if [ "$SUB_ERR" -eq 0 ]; then
    echo "  OK"
else
    ERRORS=$((ERRORS + SUB_ERR))
fi
echo ""

# --- 3. File counts per subfolder ---
echo "[3/4] Checking file counts..."
FC_ERR=0
for scene in $COMMON_SCENES; do
    for sub in $(ls -1 "${DIR_A}/${scene}/" 2>/dev/null); do
        [ -d "${DIR_A}/${scene}/${sub}" ] || continue
        FC_A=$(ls -1 "${DIR_A}/${scene}/${sub}/" 2>/dev/null | wc -l)
        FC_B=$(ls -1 "${DIR_B}/${scene}/${sub}/" 2>/dev/null | wc -l)
        if [ "$FC_A" -ne "$FC_B" ]; then
            echo "  MISMATCH [$scene/$sub]: A=$FC_A B=$FC_B"
            FC_ERR=$((FC_ERR + 1))
        fi
    done
done
if [ "$FC_ERR" -eq 0 ]; then
    echo "  OK"
else
    echo "  $FC_ERR mismatches"
    ERRORS=$((ERRORS + FC_ERR))
fi
echo ""

# --- 4. Total size per scene ---
echo "[4/4] Checking sizes per scene..."
SZ_ERR=0
for scene in $COMMON_SCENES; do
    SZ_A=$(du -sb "${DIR_A}/${scene}/" 2>/dev/null | cut -f1)
    SZ_B=$(du -sb "${DIR_B}/${scene}/" 2>/dev/null | cut -f1)
    if [ "$SZ_A" != "$SZ_B" ]; then
        HR_A=$(numfmt --to=iec "$SZ_A" 2>/dev/null || echo "${SZ_A}B")
        HR_B=$(numfmt --to=iec "$SZ_B" 2>/dev/null || echo "${SZ_B}B")
        echo "  MISMATCH [$scene]: A=$HR_A B=$HR_B"
        SZ_ERR=$((SZ_ERR + 1))
    fi
done
if [ "$SZ_ERR" -eq 0 ]; then
    echo "  OK"
else
    echo "  $SZ_ERR size mismatches"
    WARNINGS=$((WARNINGS + SZ_ERR))
fi
echo ""

# --- Summary ---
echo "=========================================="
echo "Summary: $ERRORS errors, $WARNINGS warnings"
if [ "$ERRORS" -eq 0 ] && [ "$WARNINGS" -eq 0 ]; then
    echo "RESULT: ALL CHECKS PASSED"
elif [ "$ERRORS" -eq 0 ]; then
    echo "RESULT: PASSED with warnings"
else
    echo "RESULT: FAILED"
fi
echo "=========================================="
