AdaBound_Keras
-----------

# 概要

ICLR 2019のAdaBoundという最適化手法を実装しました．
論文: https://openreview.net/pdf?id=Bkg3g2R9FX

定理２，３にあるようにAdamは最適な解に収束しないことが示せるようです．
また，(Momentum)SGDは良い解を得られるものの，収束が遅いことで知られています．

そこで，反復回数が少ない時はAdamのような振る舞いをし，徐々にモーメンタムSGDのような振る舞いをするようなAdaBoundという手法を提案したようです．

実験では，Adamより良い解を得られていることがわかります．

# 動かしてみる
https://colab.research.google.com/drive/1FKD0eXUWA3CrsQjkDovxYcojj2aPXrxI
