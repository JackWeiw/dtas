from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

m = T.int64()
[
    T.min(
        m,
        T.max(
            T.max(
                T.int64(0),
                T.int64(1) - (m + T.int64(255)) // T.int64(256) * T.int64(256),
            )
            * T.int64(2)
            + T.min(
                T.int64(0),
                (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1),
            )
            * T.int64(2)
            + m,
            m,
        )
        - T.min(
            T.max(
                T.int64(0),
                T.int64(1) - (m + T.int64(255)) // T.int64(256) * T.int64(256),
            )
            + T.min(
                T.int64(0),
                (m + T.int64(255)) // T.int64(256) * T.int64(256) - T.int64(1),
            ),
            T.int64(0),
        )
        * T.int64(3),
    )
]
