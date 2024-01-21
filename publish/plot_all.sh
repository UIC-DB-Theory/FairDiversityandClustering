python3 plot.py ./results/macro_exp_equal.json 0.2
python3 plot.py ./results/macro_exp_prop.json 0.2
python3 plot.py ./results/micro_exp_equal.json 1.2
python3 plot.py ./results/micro_exp_prop.json 1.2
python3 streamplot.py ./results/macro_stream_beer_review.json -nolegend
python3 streamplot.py ./results/macro_stream_yelp.json -nolegend