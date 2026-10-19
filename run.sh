# OPENBLAS_NUM_THREADS=1 \
# MKL_NUM_THREADS=1 \
# OMP_NUM_THREADS=1 \
# python -u main.py \
#   --experiment batch-ep \
#   --band alpha \
#   --drowsy-ids 3 \
#   --responsive-ids 2 \
#   --states baseline \
#   --models ktg \
#   --group-lasso \
#   --lambda-min-ratio 0.00001 \
#   --n-lambdas 120 \
#   --max-iter 10000 \
#   --n-jobs 64 \
#   --tol 1e-5 \
#   --lambda-solver parallel

for band in delta theta alpha beta gamma; do
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  python -u main.py \
    --experiment batch-ep \
    --band "$band" \
    --drowsy-ids 3 6 7 8 10 14 26 \
    --responsive-ids 2 5 9 13 18 20 22 23 24 25 27 28 29 \
    --states baseline mild moderate recovery \
    --models ktg artg \
    --value-type absolute \
    --group-lasso \
    --lambda-min-ratio 0.00001 \
    --n-lambdas 120 \
    --max-iter 10000 \
    --n-jobs 64 \
    --tol 1e-5 \
    --lambda-solver parallel
done