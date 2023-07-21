# vary the size for the cast_first case
memray run -o cast_first_10_1e6 run_test.py 10 1000000 cast_first
memray summary cast_first_10_1e6 > cast_first_10_1e6.summary
memray stats cast_first_10_1e6 > cast_first_10_1e6.stats
python -m timeit -s "from run_test import cast_first" -n 10 "cast_first(10, 1000000)" > cast_first_10_1e6.timeit

memray run -o cast_first_10_1e7 run_test.py 10 10000000 cast_first
memray summary cast_first_10_1e7 > cast_first_10_1e7.summary
memray stats cast_first_10_1e7 > cast_first_10_1e7.stats
python -m timeit -s "from run_test import cast_first" -n 10 "cast_first(10, 10000000)" > cast_first_10_1e7.timeit

memray run -o cast_first_10_1e8 run_test.py 10 100000000 cast_first
memray summary cast_first_10_1e8 > cast_first_10_1e8.summary
memray stats cast_first_10_1e8 > cast_first_10_1e8.stats
python -m timeit -s "from run_test import cast_first" -n 10 "cast_first(10, 100000000)" > cast_first_10_1e8.timeit

# vary the size of the cast_last case
memray run -o cast_last_10_1e6 run_test.py 10 1000000 cast_last
memray summary cast_last_10_1e6 > cast_last_10_1e6.summary
memray stats cast_last_10_1e6 > cast_last_10_1e6.stats
python -m timeit -s "from run_test import cast_last" -n 10 "cast_last(10, 1000000)" > cast_last_10_1e6.timeit

memray run -o cast_last_10_1e7 run_test.py 10 10000000 cast_last
memray summary cast_last_10_1e7 > cast_last_10_1e7.summary
memray stats cast_last_10_1e7 > cast_last_10_1e7.stats
python -m timeit -s "from run_test import cast_last" -n 10 "cast_last(10, 10000000)" > cast_last_10_1e7.timeit

memray run -o cast_last_10_1e8 run_test.py 10 100000000 cast_last
memray summary cast_last_10_1e8 > cast_last_10_1e8.summary
memray stats cast_last_10_1e8 > cast_last_10_1e8.stats
python -m timeit -s "from run_test import cast_last" -n 10 "cast_last(10, 100000000)" > cast_last_10_1e8.timeit

# do some runs with different base types and no type cast. concat only
memray run -o no_cast_float64 run_test.py 10 100000000 no_cast --basetype float64
memray summary no_cast_float64 > no_cast_float64.summary
memray stats no_cast_float64 > no_cast_float64.stats
python -m timeit -s "from run_test import no_cast" -n 10 "no_cast(10, 100000000)" > no_cast_float64.timeit

memray run -o no_cast_float16 run_test.py 10 100000000 no_cast --basetype float16
memray summary no_cast_float16 > no_cast_float16.summary
memray stats no_cast_float16 > no_cast_float16.stats
python -m timeit -s "from run_test import no_cast" -n 10 "no_cast(10, 100000000)" > no_cast_float16.timeit

memray run -o no_cast_int run_test.py 10 100000000 no_cast --basetype int
memray summary no_cast_int > no_cast_int.summary
memray stats no_cast_int > no_cast_int.stats
python -m timeit -s "from run_test import no_cast" -n 10 "no_cast(10, 100000000)" > no_cast_int.timeit

# vary the base dtype for cast first and last 
memray run -o cast_first_from_float32 run_test.py 10 100000000 cast_first --base_type float32
memray summary cast_first_from_float32 > cast_first_from_float32.summary
memray stats cast_first_from_float32 > cast_first_from_float32.stats
python -m timeit -s "from run_test import cast_first" -n 10 "cast_first(10, 100000000, base_type='float32')" > cast_first_from_float32.timeit

memray run -o cast_last_from_float32 run_test.py 10 100000000 cast_last --base_type float32
memray summary cast_last_from_float32 > cast_last_from_float32.summary
memray stats cast_last_from_float32 > cast_last_from_float32.stats
python -m timeit -s "from run_test import cast_last" -n 10 "cast_last(10, 100000000, base_type='float32')" > cast_last_from_float32.timeit

memray run -o cast_first_from_float16 run_test.py 10 100000000 cast_first --base_type float16
memray summary cast_first_from_float16 > cast_first_from_float16.summary
memray stats cast_first_from_float16 > cast_first_from_float16.stats
python -m timeit -s "from run_test import cast_first" -n 10 "cast_first(10, 100000000, base_type='float16')" > cast_first_from_float16.timeit

memray run -o cast_last_from_float16 run_test.py 10 100000000 cast_last --base_type float16
memray summary cast_last_from_float16 > cast_last_from_float16.summary
memray stats cast_last_from_float16 > cast_last_from_float16.stats
python -m timeit -s "from run_test import cast_last" -n 10 "cast_last(10, 100000000, base_type='float16')" > cast_last_from_float16.timeit

mv cast_* results/
mv no_cast* results/
