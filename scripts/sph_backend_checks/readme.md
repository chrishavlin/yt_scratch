instructions: 

for both the `main` yt branch and PR branch at https://github.com/yt-project/yt/pull/4939

1. checkout the main branch 
2. pip install -e . 
3. run `python res_tests.py main`
4. checkout branch from https://github.com/yt-project/yt/pull/4939
5. pip install -3 . 
6. run `python res_tests.py sph_proj_backend`
7. run `python plot_results.py` 

This will generate png and json files for each branch and a comparison plot between branches 
