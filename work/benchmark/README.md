The modified benchmark project sources are placed in this directory.

The original benchamrk project is : https://github.com/OdysseasKr/neural-disaggregator

The implementations do not include feature engineering, instead, they are using only total electricity conunsuption data for training.

However, we proposed a unique feature engineering, so it can be applied to the benchmark project's NILM models.

This directory includes the following 5 modified NILM models with the proposed feature engineering.  

- daedisaggregator_fe.py
- grudisaggregator_fe.py
- rnndisaggregator_fe.py
- shortseq2pointdisaggregator_fe.py
- windowgrudisaggregator_fe.py

postfix "_fe" means feature engineering.
